cat ./checkpoints_tar_parts/checkpoints.tar.bz2.part* > checkpoints.tar.bz2
tar -xvf ../checkpoints.tar.bz2
rm checkpoints.tar.bz2
