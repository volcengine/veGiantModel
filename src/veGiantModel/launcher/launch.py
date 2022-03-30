# Copyright (c) 2021, ByteDance Inc.  All rights reserved.
#!/usr/bin/python

from __future__ import print_function
import os
import subprocess
import threading
import sys
from megatron import mpu
from deepspeed.utils import log_dist
import logging

class PropagatingThread(threading.Thread):
    """ propagate exceptions to the parent's thread
    refer to https://stackoverflow.com/a/31614591/9601110
    """

    def run(self):
        self.exc = None
        try:
            if hasattr(self, '_Thread__target'):
                #  python 2.x
                self.ret = self._Thread__target(
                    *self._Thread__args, **self._Thread__kwargs)
            else:
                # python 3.x
                self.ret = self._target(*self._args, **self._kwargs)
        except BaseException as e:
            self.exc = e

    def join(self):
        super(PropagatingThread, self).join()
        if self.exc:
            raise self.exc
        return self.exc

def launch_scheduler(local_rank):
    if os.environ['WORKER_RANK'] != '0':
        return

    if local_rank != 0:
        return

    def scheduler_runner():
        my_env = os.environ.copy()
        my_env['DMLC_ROLE'] = 'scheduler'
        my_env['PS_VERBOSE'] = os.environ.get('PS_VERBOSE', '1')
        nvidia_smi = f'nvidia-smi -L'
        devices = os.popen(nvidia_smi).read().strip()
        if 'A100' in devices:
            my_env['DMLC_NODE_HOST'] = get_worker0_host()
            my_env['UCX_RDMA_CM_SOURCE_ADDRESS'] = get_worker0_host()
            my_env['DMLC_INTERFACE'] = os.environ.get('VE_GIANT_MODEL_SCHED_INTERFACE', 'eth1')
            dmlc_interface = my_env['DMLC_INTERFACE']
            os.environ['UCX_NET_DEVICES'] = 'mlx5_2:1,eth0,eth1,eth2,eth3'
            log_dist(f'scheduler DMLC_NODE_HOST: {get_worker0_host()}, DMLC_INTERFACE:{dmlc_interface}', ranks=[-1])

        command = "python3 -c 'import byteps.server'"
        subprocess.check_call(command, env=my_env,
                          stdout=sys.stdout, stderr=sys.stderr, shell=True)
    t = PropagatingThread(target=scheduler_runner)
    t.setDaemon(True)
    t.start()

def get_worker0_host():
    host = os.environ['WORKER_0_HOST']
    return host

def get_worker0_port():
    port = os.environ['WORKER_0_PORT']
    return port

def get_nic(local_rank):

    nic_cmd1 = 'nvidia-smi topo -m | grep mlx | grep PIX | wc -l'
    nic_cmd2 = 'nvidia-smi topo -m | grep mlx | grep PXB | wc -l'
    nic_count1 = int(os.popen(nic_cmd1).read().strip())
    nic_count2 = int(os.popen(nic_cmd2).read().strip())

    nic_count = max(nic_count1, nic_count2)
    log_dist(f'get_nic:nic_count={nic_count}, nic_count1:{nic_count1}, nic_count2:{nic_count2}', ranks=[-1])

    dmlc_nic_offset = int(os.environ.get('VE_GIANT_MODEL_NIC_OFFSET', 1))
    if nic_count == 0:
        nic_get = 1
    elif nic_count == 2:
        nic_get = int(local_rank / 4) + dmlc_nic_offset
    elif nic_count == 4:
        nic_get = int(local_rank / 2) + dmlc_nic_offset
    else:
        nic_get = 1
    nic = os.environ.get('VE_GIANT_MODEL_NIC_BIND', nic_get)
    os.environ['DMLC_INTERFACE'] = f'eth{nic}'
    log_dist(f'DMLC_INTERFACE: eth{nic}', ranks=[-1])
    return nic

def setup_env(local_rank):
    mp_size = mpu.get_model_parallel_world_size()

    num_nodes = int(os.environ['NUM_WORKER'])
    gpu_per_node = int(os.environ['GPU_PER_WORKER'])
    assert gpu_per_node >= mp_size
    assert gpu_per_node % mp_size == 0

    os.environ['BYTEPS_RDMA_START_DEPTH'] = str(32)
    os.environ['BYTEPS_RDMA_RX_DEPTH'] = str(512)

    os.environ['DMLC_NUM_WORKER'] = str(gpu_per_node * num_nodes)
    os.environ['DMLC_NUM_SERVER'] = str(gpu_per_node * num_nodes)

    os.environ['BYTEPS_LOCAL_SIZE'] = str(gpu_per_node)
    os.environ['BYTEPS_FORCE_DISTRIBUTED'] = '1'
    os.environ['BYTEPS_ENABLE_IPC'] = '0'
    os.environ['DMLC_PS_ROOT_PORT'] = get_worker0_port()
    os.environ['DMLC_PS_ROOT_URI'] = get_worker0_host()

    os.environ['DMLC_ENABLE_RDMA'] = os.environ.get('DMLC_ENABLE_RDMA', '1')
    os.environ['DMLC_ENABLE_UCX'] = os.environ.get('DMLC_ENABLE_UCX', '1')
    os.environ['UCX_IB_TRAFFIC_CLASS'] = '236'
    os.environ['UCX_TLS'] = os.environ.get('UCX_TLS', 'rc_x,tcp,sm')
    nvidia_smi = f'nvidia-smi -L'
    devices = os.popen(nvidia_smi).read().strip()
    if 'A100' in devices:
        nic = get_nic(local_rank)
        ip_cmd = f'ip addr show eth{nic}'
        ip = os.popen(ip_cmd + ' | grep "\<inet\>" | awk \'{ print $2 }\' | awk -F "/" \'{ print $1 }\'').read().strip()
        os.environ['UCX_RDMA_CM_SOURCE_ADDRESS'] = os.environ.get('UCX_RDMA_CM_SOURCE_ADDRESS', ip)
        devs = os.environ.get('UCX_NET_DEVICES', f'mlx5_{nic}:1,eth0,eth1,eth2,eth3')
        os.environ['UCX_NET_DEVICES'] = devs
        os.environ['DMLC_NODE_HOST'] = os.environ['UCX_RDMA_CM_SOURCE_ADDRESS']
    elif 'V100' in devices or 'T4' in devices:
        devs = os.environ.get('UCX_NET_DEVICES', 'mlx5_2:1,eth0,eth2')
        os.environ['UCX_NET_DEVICES'] = devs
    else:
        raise RuntimeError(f"Unknown devices: {devices}")

def launch_bps(local_rank):
    log_dist(f'launch_bps({local_rank})', ranks=[-1], level=logging.DEBUG)
    setup_env(local_rank)
    launch_scheduler(local_rank)