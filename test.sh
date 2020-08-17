set -ex

NUM_CORES=1 PYTHONPATH=src exec python3pdb src/generate_test.py --length 16 --batch_size 4 "$@"
