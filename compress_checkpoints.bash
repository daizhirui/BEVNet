sudo apt install pbzip2 -y
mkdir -p checkpoints_tar_parts
cd checkpoints_tar_parts || return  # stop on cd failure
rm ./*
tar --use-compress-program=pbzip2 -cvf -  ../checkpoints | split -b 45M - "checkpoints.tar.bz2.part"
