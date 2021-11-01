# Standard Library
from collections import defaultdict
from typing import List
import unittest

# Thirdparty libraries
from nvgpu import gpu_info


class TestGPUInfo(unittest.TestCase):

    def test_gpu_info(self):
        for name, test_case in test_cases.items():
            with self.subTest(name=name):
                infos = gpu_info(test_case)

            self.assertEqual(len(test_case.infos), len(infos))

            for true_info, retrieved_info in zip(test_case.infos, infos):
                for k, v in true_info.items():
                    with self.subTest(info_type=k):
                        self.assertIn(k, retrieved_info)
                        self.assertEqual(v, retrieved_info.get(k, None))


class NvidiaSMIOutput:
    nvidia_smi_L: List[str]
    nvidia_smi: List[str]
    infos: List[dict] = [dict()]


test_cases = defaultdict(NvidiaSMIOutput)

########################################################################################################################

test_cases['sample_cuda_114'].nvidia_smi_L = """GPU 0: NVIDIA GeForce RTX 3090 (UUID: GPU-4738f547-e85e-3da4-ea09-dceea83a3afb)""".split(
    '\n')
test_cases['sample_cuda_114'].nvidia_smi = """Mon Nov  1 18:28:16 2021
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.63.01    Driver Version: 470.63.01    CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:2D:00.0 Off |                  N/A |
|  0%   46C    P8    28W / 350W |    139MiB / 24267MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      1197      G   /usr/lib/xorg/Xorg                100MiB |
|    0   N/A  N/A      1293      G   /usr/bin/gnome-shell               37MiB |
+-----------------------------------------------------------------------------+""".split('\n')
test_cases['sample_cuda_114'].infos = [dict(
    index='0',
    type='NVIDIA GeForce RTX 3090',
    uuid='GPU-4738f547-e85e-3da4-ea09-dceea83a3afb',
    mem_used=139,
    mem_total=24267,
    mem_used_percent=0.5727943297482178,
)]

########################################################################################################################

test_cases['sample_cuda_11_mig'].nvidia_smi_L = """GPU 0: A100-SXM4-40GB (UUID: GPU-2ac434d3-1aca-57a0-597d-1d2effdba9f3)
  MIG 3g.20gb Device 0: (UUID: MIG-GPU-2ac434d3-1aca-57a0-597d-1d2effdba9f3/1/0)
  MIG 3g.20gb Device 1: (UUID: MIG-GPU-2ac434d3-1aca-57a0-597d-1d2effdba9f3/2/0)
GPU 1: A100-SXM4-40GB (UUID: GPU-cd9ee947-34d2-d0fe-93e0-2195b3f2fd52)
  MIG 3g.20gb Device 0: (UUID: MIG-GPU-cd9ee947-34d2-d0fe-93e0-2195b3f2fd52/1/0)
  MIG 3g.20gb Device 1: (UUID: MIG-GPU-cd9ee947-34d2-d0fe-93e0-2195b3f2fd52/2/0)
GPU 2: A100-SXM4-40GB (UUID: GPU-67051dc2-cce1-4acf-d254-e76d938eda30)
  MIG 7g.40gb Device 0: (UUID: MIG-GPU-67051dc2-cce1-4acf-d254-e76d938eda30/0/0)
GPU 3: A100-SXM4-40GB (UUID: GPU-30f9801c-a554-b6de-bd27-d5cefbf59291)
  MIG 7g.40gb Device 0: (UUID: MIG-GPU-30f9801c-a554-b6de-bd27-d5cefbf59291/0/0)""".split('\n')

test_cases['sample_cuda_11_mig'].nvidia_smi = """Mon Nov  1 18:03:37 2021
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 450.142.00   Driver Version: 450.142.00   CUDA Version: 11.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  A100-SXM4-40GB      Off  | 00000000:01:00.0 Off |                   On |
| N/A   46C    P0   127W / 400W |  16225MiB / 40537MiB |     N/A      Default |
|                               |                      |              Enabled |
+-------------------------------+----------------------+----------------------+
|   1  A100-SXM4-40GB      Off  | 00000000:41:00.0 Off |                   On |
| N/A   42C    P0    74W / 400W |  15103MiB / 40537MiB |     N/A      Default |
|                               |                      |              Enabled |
+-------------------------------+----------------------+----------------------+
|   2  A100-SXM4-40GB      Off  | 00000000:81:00.0 Off |                   On |
| N/A   45C    P0   259W / 400W |  15356MiB / 40537MiB |     N/A      Default |
|                               |                      |              Enabled |
+-------------------------------+----------------------+----------------------+
|   3  A100-SXM4-40GB      Off  | 00000000:C1:00.0 Off |                   On |
| N/A   45C    P0   232W / 400W |  21156MiB / 40537MiB |     N/A      Default |
|                               |                      |              Enabled |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| MIG devices:                                                                |
+------------------+----------------------+-----------+-----------------------+
| GPU  GI  CI  MIG |         Memory-Usage |        Vol|         Shared        |
|      ID  ID  Dev |           BAR1-Usage | SM     Unc| CE  ENC  DEC  OFA  JPG|
|                  |                      |        ECC|                       |
|==================+======================+===========+=======================|
|  0    1   0   0  |  16214MiB / 20096MiB | 42      0 |  3   0    2    0    0 |
|                  |      4MiB / 32767MiB |           |                       |
+------------------+----------------------+-----------+-----------------------+
|  0    2   0   1  |     11MiB / 20096MiB | 42      0 |  3   0    2    0    0 |
|                  |      0MiB / 32767MiB |           |                       |
+------------------+----------------------+-----------+-----------------------+
|  1    1   0   0  |     11MiB / 20096MiB | 42      0 |  3   0    2    0    0 |
|                  |      0MiB / 32767MiB |           |                       |
+------------------+----------------------+-----------+-----------------------+
|  1    2   0   1  |  15092MiB / 20096MiB | 42      0 |  3   0    2    0    0 |
|                  |      4MiB / 32767MiB |           |                       |
+------------------+----------------------+-----------+-----------------------+
|  2    0   0   0  |  15356MiB / 40537MiB | 98      0 |  7   0    5    1    1 |
|                  |      7MiB / 65536MiB |           |                       |
+------------------+----------------------+-----------+-----------------------+
|  3    0   0   0  |  21156MiB / 40537MiB | 98      0 |  7   0    5    1    1 |
|                  |     15MiB / 65536MiB |           |                       |
+------------------+----------------------+-----------+-----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0    1    0      53522      C   python3                         16197MiB |
|    1    2    0     111478      C   python                          15077MiB |
|    2    0    0       1128      C   python                          15351MiB |
|    3    0    0      10179      C   python3                         21149MiB |
+-----------------------------------------------------------------------------+""".split('\n')

test_cases['sample_cuda_11_mig'].infos = [
    dict(
        index='0',
        type='A100-SXM4-40GB',
        uuid='GPU-2ac434d3-1aca-57a0-597d-1d2effdba9f3',
        mem_used=16225,
        mem_total=40537,
        mem_used_percent=100. * 16225 / 40537,
    ),
    dict(
        index='1',
        type='A100-SXM4-40GB',
        uuid='GPU-cd9ee947-34d2-d0fe-93e0-2195b3f2fd52',
        mem_used=15103,
        mem_total=40537,
        mem_used_percent=100. * 15103 / 40537,
    ),
    dict(
        index='2',
        type='A100-SXM4-40GB',
        uuid='GPU-67051dc2-cce1-4acf-d254-e76d938eda30',
        mem_used=15356,
        mem_total=40537,
        mem_used_percent=100. * 15356 / 40537,
    ),
    dict(
        index='3',
        type='A100-SXM4-40GB',
        uuid='GPU-30f9801c-a554-b6de-bd27-d5cefbf59291',
        mem_used=21156,
        mem_total=40537,
        mem_used_percent=100. * 21156 / 40537,
    ),
]
