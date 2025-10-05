"""
OptiTrack NatNet → ZMQ bridge.

Receives rigid body frames from an OptiTrack NatNetClient and publishes
poses via ZeroMQ PUB so they can be consumed by sim2real/tpg/tpg_ext_main.py.

Message format (single pyobj per publish):
    {
      "go2_0": [[x, y, z], [w, x, y, z]],
      "go1_1": [[x, y, z], [w, x, y, z]],
      ...
    }

Notes
- The tpg runner expects dict keys to end with the agent index (e.g., _0, _1).
- Quaternion ordering is [w, x, y, z]. If NatNet provides [x, y, z, w], it is re-ordered.
- You can pass an explicit id→name mapping via CLI (e.g., --map 1:go2_0,2:go1_1).
"""

import sys
import time
import argparse
import threading
from typing import Dict, Tuple, List

import zmq

try:
    from natnet.NatNetClient import NatNetClient
except Exception as e:  # pragma: no cover
    raise RuntimeError("NatNetClient import failed. Ensure natnet SDK is installed.") from e


def xyzw_to_wxyz(q: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x, y, z, w = q
    return (w, x, y, z)


class OptiTrackZMQBridge:
    def __init__(
        self,
        client_ip: str,
        server_ip: str,
        use_multicast: bool,
        zmq_pub_port: int,
        id_to_agent: Dict[int, str],
        publish_hz: float = 200.0,
        assume_xyzw_quat: bool = True,
    ) -> None:
        self.id_to_agent = id_to_agent
        self.assume_xyzw_quat = assume_xyzw_quat
        self.publish_period = 1.0 / max(1.0, publish_hz)

        # Shared pose dict: {agent_name: ([x,y,z], [w,x,y,z])}
        self._pose_dict: Dict[str, Tuple[List[float], List[float]]] = {}
        self._pose_lock = threading.Lock()

        # ZMQ PUB socket
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{zmq_pub_port}")
        print(f"OptiTrackZMQBridge: Publishing mocap on tcp://*:{zmq_pub_port}")

        # NatNet client
        self.client = NatNetClient()
        self.client.set_client_address(client_ip)
        self.client.set_server_address(server_ip)
        self.client.set_use_multicast(use_multicast)
        # Listener for RB frames
        self.client.rigid_body_listener = self._on_rigid_body_frame

    def start(self) -> None:
        # Start NatNet streaming on background thread
        if not self.client.run():
            raise RuntimeError("Could not start NatNet streaming client")
        time.sleep(0.5)
        if not self.client.connected():
            raise RuntimeError("NatNet client failed to connect. Check Motive streaming.")
        print("OptiTrackZMQBridge: NatNet connected")

        # Start publisher thread
        t = threading.Thread(target=self._publish_loop, daemon=True)
        t.start()

        print("OptiTrackZMQBridge: Running… Ctrl+C to stop")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
        finally:
            self.client.shutdown()

    def _on_rigid_body_frame(self, rb_id, position, rotation, tracking_valid=True, mrk=None):
        """NatNet rigid body callback.

        Expected signature aligns with NatNet Python samples:
          - rb_id: int
          - position: (x, y, z)
          - rotation: quaternion, typically (x, y, z, w)
        Additional args are tolerated for compatibility across versions.
        """
        try:
            agent_name = self.id_to_agent.get(int(rb_id))
            if agent_name is None:
                return  # Not interested in this RB id

            # Position -> list
            px, py, pz = position
            pos = [float(px), float(py), float(pz)]

            # Quaternion -> [w,x,y,z]
            if self.assume_xyzw_quat:
                quat_wxyz = list(xyzw_to_wxyz(rotation))
            else:
                # Already [w,x,y,z]
                qw, qx, qy, qz = rotation
                quat_wxyz = [float(qw), float(qx), float(qy), float(qz)]

            with self._pose_lock:
                self._pose_dict[agent_name] = (pos, quat_wxyz)
        except Exception as e:
            # Be robust to any unexpected data shape; do not spam logs
            pass

    @staticmethod
    def _sorted_agents(d: Dict[str, Tuple[List[float], List[float]]]) -> List[Tuple[str, Tuple[List[float], List[float]]]]:
        def idx_from_name(name: str) -> int:
            try:
                return int(name.split("_")[-1])
            except Exception:
                return 1_000_000

        return sorted(d.items(), key=lambda kv: idx_from_name(kv[0]))

    def _publish_loop(self) -> None:
        last_pub = 0.0
        while True:
            now = time.time()
            if now - last_pub < self.publish_period:
                time.sleep(0.001)
                continue

            with self._pose_lock:
                # Build an ordered dict-like plain dict to preserve insertion order
                ordered = {}
                for name, (pos, quat) in self._sorted_agents(self._pose_dict):
                    ordered[name] = [pos, quat]

            if ordered:
                try:
                    self.socket.send_pyobj(ordered)
                except Exception:
                    pass

            last_pub = now


def parse_id_map(map_str: str) -> Dict[int, str]:
    """Parse id:name pairs like "1:go2_0,2:go1_1" into a dict."""
    result = {}
    if not map_str:
        return result
    for token in map_str.split(","):
        if not token:
            continue
        if ":" not in token:
            continue
        k, v = token.split(":", 1)
        try:
            result[int(k.strip())] = v.strip()
        except ValueError:
            continue
    return result


def main():
    parser = argparse.ArgumentParser(description="OptiTrack NatNet → ZMQ bridge")
    parser.add_argument("--client", dest="client_ip", type=str, default="192.168.1.3")
    parser.add_argument("--server", dest="server_ip", type=str, default="192.168.1.1")
    parser.add_argument("--unicast", dest="use_multicast", action="store_false", help="Use unicast (default)")
    parser.add_argument("--multicast", dest="use_multicast", action="store_true", help="Use multicast")
    parser.set_defaults(use_multicast=False)
    parser.add_argument("--pub_port", type=int, default=6000, help="ZMQ PUB port")
    parser.add_argument(
        "--map",
        type=str,
        default="",
        help="RigidBodyID→agent mapping, e.g., '1:go2_0,2:go1_1'",
    )
    parser.add_argument(
        "--hz", type=float, default=200.0, help="Publish frequency (Hz)"
    )
    parser.add_argument(
        "--quat_xyzw", action="store_true", help="NatNet rotation is [x,y,z,w] (default)"
    )
    parser.add_argument(
        "--quat_wxyz", action="store_true", help="NatNet rotation already [w,x,y,z]"
    )
    args = parser.parse_args()

    id_to_agent = parse_id_map(args.map)
    if not id_to_agent:
        print(
            "Warning: no id→agent mapping provided. Use --map 1:go2_0,2:go1_1 to target specific rigid bodies."
        )

    bridge = OptiTrackZMQBridge(
        client_ip=args.client_ip,
        server_ip=args.server_ip,
        use_multicast=args.use_multicast,
        zmq_pub_port=args.pub_port,
        id_to_agent=id_to_agent,
        publish_hz=args.hz,
        assume_xyzw_quat=not args.quat_wxyz,
    )
    bridge.start()


if __name__ == "__main__":
    main()
