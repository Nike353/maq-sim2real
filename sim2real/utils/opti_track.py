# OptiTrack NatNet direct depacketization sample for Python 3.x
#
# Uses the Python NatNetClient.py library to establish a connection (by creating a NatNetClient),
# and receive data via a NatNet connection and decode it using the NatNetClient library.

import sys
import time

# sys.path.append("/home/shiqi/Documents/Mocap")

from natnet.NatNetClient import NatNetClient
from natnet import DataDescriptions
from natnet import MoCapData


from scripts.utilities import (
    my_parse_args,
    receive_new_frame,
    receive_rigid_body_frame,
    print_commands,
    print_configuration,
    request_data_descriptions,
)


if __name__ == "__main__":

    optionsDict = {}
    optionsDict["clientAddress"] = "192.168.1.3"
    optionsDict["serverAddress"] = "192.168.1.1"
    optionsDict["use_multicast"] = False

    # This will create a new NatNet client
    optionsDict = my_parse_args(sys.argv, optionsDict)

    streaming_client = NatNetClient()
    streaming_client.set_client_address(optionsDict["clientAddress"])
    streaming_client.set_server_address(optionsDict["serverAddress"])
    streaming_client.set_use_multicast(optionsDict["use_multicast"])

    # Configure the streaming client to call our rigid body handler on the emulator to send data out.
    streaming_client.new_frame_listener = receive_new_frame
    streaming_client.rigid_body_listener = receive_rigid_body_frame

    # Start up the streaming client now that the callbacks are set up.
    # This will run perpetually, and operate on a separate thread.
    is_running = streaming_client.run()
    if not is_running:
        print("ERROR: Could not start streaming client.")
        try:
            sys.exit(1)
        except SystemExit:
            print("...")
        finally:
            print("exiting")

    is_looping = True
    time.sleep(1)
    if streaming_client.connected() is False:
        print("ERROR: Could not connect properly.  Check that Motive streaming is on.")
        try:
            sys.exit(2)
        except SystemExit:
            print("...")
        finally:
            print("exiting")

    print_configuration(streaming_client)
    print("\n")
    print_commands(streaming_client.can_change_bitstream_version())

    while is_looping:
        inchars = input("Enter command or ('h' for list of commands)\n")
        if len(inchars) > 0:
            c1 = inchars[0].lower()
            if c1 == "h":
                print_commands(streaming_client.can_change_bitstream_version())
            elif c1 == "c":
                print_configuration(streaming_client)
            elif c1 == "s":
                request_data_descriptions(streaming_client)
                time.sleep(1)
            elif (c1 == "3") or (c1 == "4"):
                if streaming_client.can_change_bitstream_version():
                    tmp_major = 4
                    tmp_minor = 1
                    if c1 == "3":
                        tmp_major = 3
                        tmp_minor = 1
                    return_code = streaming_client.set_nat_net_version(
                        tmp_major, tmp_minor
                    )
                    time.sleep(1)
                    if return_code == -1:
                        print(
                            "Could not change bitstream version to %d.%d"
                            % (tmp_major, tmp_minor)
                        )
                    else:
                        print("Bitstream version at %d.%d" % (tmp_major, tmp_minor))
                else:
                    print("Can only change bitstream in Unicast Mode")

            elif c1 == "p":
                sz_command = "TimelineStop"
                return_code = streaming_client.send_command(sz_command)
                time.sleep(1)
                print("Command: %s - return_code: %d" % (sz_command, return_code))
            elif c1 == "r":
                sz_command = "TimelinePlay"
                return_code = streaming_client.send_command(sz_command)
                print("Command: %s - return_code: %d" % (sz_command, return_code))
            elif c1 == "o":
                tmpCommands = [
                    "TimelinePlay",
                    "TimelineStop",
                    "SetPlaybackStartFrame,0",
                    "SetPlaybackStopFrame,1000000",
                    "SetPlaybackLooping,0",
                    "SetPlaybackCurrentFrame,0",
                    "TimelineStop",
                ]
                for sz_command in tmpCommands:
                    return_code = streaming_client.send_command(sz_command)
                    print("Command: %s - return_code: %d" % (sz_command, return_code))
                time.sleep(1)
            elif c1 == "w":
                tmp_commands = [
                    "TimelinePlay",
                    "TimelineStop",
                    "SetPlaybackStartFrame,1",
                    "SetPlaybackStopFrame,1500",
                    "SetPlaybackLooping,0",
                    "SetPlaybackCurrentFrame,100",
                    "TimelineStop",
                ]
                for sz_command in tmp_commands:
                    return_code = streaming_client.send_command(sz_command)
                    print("Command: %s - return_code: %d" % (sz_command, return_code))
                time.sleep(1)
            elif c1 == "t":
                test_classes()

            elif c1 == "j":
                streaming_client.set_print_level(0)
                print(
                    "Showing only received frame numbers and supressing data descriptions"
                )
            elif c1 == "k":
                streaming_client.set_print_level(1)
                print("Showing every received frame")

            elif c1 == "l":
                print_level = streaming_client.set_print_level(20)
                print_level_mod = print_level % 100
                if print_level == 0:
                    print(
                        "Showing only received frame numbers and supressing data descriptions"
                    )
                elif print_level == 1:
                    print("Showing every frame")
                elif print_level_mod == 1:
                    print("Showing every %dst frame" % print_level)
                elif print_level_mod == 2:
                    print("Showing every %dnd frame" % print_level)
                elif print_level == 3:
                    print("Showing every %drd frame" % print_level)
                else:
                    print("Showing every %dth frame" % print_level)

            elif c1 == "q":
                is_looping = False
                streaming_client.shutdown()
                break
            else:
                print("Error: Command %s not recognized" % c1)
            print("Ready...\n")
    print("exiting")