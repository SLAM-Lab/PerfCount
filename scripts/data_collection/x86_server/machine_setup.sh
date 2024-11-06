# Start TMUX Session
tmux

# Navigate to SPEC DIR
cd $SPEC_DIR
source shrc

# Build SPEC with 1 cpu config
runcpu --config=matthew-1cpu --action=build fprate intrate

# Set CPU frequency to 1.5GHz
sudo cpupower frequency-set -u 1.5GHz -d 1.5GHz -g performance

# Navigate back to data collection script location
cd $SCRIPT_DIR

# Launch collection and exit machine
./collection.sh &
exit
exit
