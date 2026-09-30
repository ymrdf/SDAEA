import array
import json
import os
import socket
import unittest

from gpu_vision_transport import GpuVisionTransport
import test_fast_godot_env as protocol


class GPUTransportTests(unittest.TestCase):
    def test_fragmented_handshake_closes_received_descriptors(self):
        sender, receiver = socket.socketpair()
        fd = os.open('/dev/null', os.O_RDONLY)
        received = []
        def construct(connection, metadata, fds):
            self.assertEqual(metadata['agent_index'], 7)
            for value in fds:
                os.fstat(value)
            received.extend(fds)
            return 'transport'
        try:
            payload = json.dumps(dict(agent_index=7)).encode()
            sender.sendmsg([payload[:5]], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array('i', [fd, fd]))])
            sender.sendall(payload[5:]); sender.shutdown(socket.SHUT_WR)
            result = GpuVisionTransport.receive.__func__(construct, receiver)
            self.assertEqual(result, 'transport')
            for value in received:
                with self.assertRaises(OSError): os.fstat(value)
        finally:
            sender.close(); receiver.close(); os.close(fd)

    def transports(self):
        result = {}
        for index in range(2):
            transport = GpuVisionTransport.__new__(GpuVisionTransport)
            transport.last_frame_id = None
            transport.copy_pair = lambda i=index: (f'left{i}', f'right{i}')
            result[index] = transport
        return result

    def test_same_frame_routes_each_robot_and_rejects_replay(self):
        observations = [dict(gpu_rgb=True, gpu_agent_index=i, gpu_frame_id=10) for i in range(2)]
        payload = json.dumps(dict(type='observe', physics_frame=4, obs=observations)).encode()
        env = protocol.ProtocolTests().env(payload)
        env.gpu_transports = self.transports(); env.gpu_transport = env.gpu_transports[0]
        response = env._get_json_dict()
        self.assertEqual([o['left_eye'] for o in response['obs']], ['left0', 'left1'])
        self.assertEqual(env.rgb_frames, 1)
        self.assertEqual(env.bytes_received, len(payload)+4)
        with self.assertRaises(RuntimeError):
            env.gpu_transports[0].attach_observation({}, 10)

    def test_swapped_agent_id_rejected(self):
        payload = json.dumps(dict(type='observe', obs=[dict(gpu_rgb=True,gpu_agent_index=1,gpu_frame_id=1)])).encode()
        env = protocol.ProtocolTests().env(payload)
        env.gpu_transports = self.transports();env.gpu_transport=env.gpu_transports[0]
        with self.assertRaises(ValueError): env._get_json_dict()


if __name__ == '__main__': unittest.main()
