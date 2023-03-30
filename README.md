# MLP_CW4_Feature_based_KD
This Feature-based KD is based on the Torchdistill Repo

The training log for teacher model is stored as log/cifar100/ce/resnet44-teacher-equal_split.log.

The training logs for student model are stored in log/cifar100/fitnet/

The trained teacher model is stored as resource/ckpt/cifar100/ce/cifar100-resnet44-teacher.pt.

The trained student models are stored in resource/ckpt/cifar100/fitnet/.

The Torchdistill directory is modified based on the original implement @https://github.com/yoshitomo-matsubara/torchdistill/tree/main/torchdistill

The yaml file for training teacher is stored as torchdistill/configs/sample/cifar100/ce/resnet44-teacher.yaml.
The yaml file for training student is stored in torchdistill/configs/sample/cifar100/fitnet.
