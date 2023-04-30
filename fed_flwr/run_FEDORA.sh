#!/bin/bash
set -e

# initialize conda env
source activate fed

# start server, wait before launching clients
python rl_server.py &
sleep 3

# start clients
for i in `seq 1 5`; do
    echo "Starting client $i"
    python rl_client.py --gpu-index 0 --eval-env hopper-expert-v2 \
        --start-index $(((i-1)*5000))  --stop-index $((i*5000)) &
done

for i in `seq 6 10`; do
    echo "Starting client $i"
    python rl_client.py --gpu-index 1 --eval-env hopper-medium-v2 \
        --start-index $(((i-1)*5000))  --stop-index $((i*5000)) &
done

# enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM

# wait for all background processes to complete
wait