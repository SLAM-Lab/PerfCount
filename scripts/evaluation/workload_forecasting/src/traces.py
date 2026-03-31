# Utility functions to read and write traces
import pandas as pd
import os

DEFAULT_FOLDER = '/home/meb4744/PerfCount/data/arm_server/final_csvs'
class benchmark_name:
	def __init__(self, name, number, refs):
		self.name = name
		self.number = number
		self.refs = refs

BENCHMARKS = {  'perlbench' : benchmark_name('perlbench_s', '600', ['0', '1', '2']),
				'mcf'       : benchmark_name('mcf_s',       '605', ['0']),
				'cactuBSSN' : benchmark_name('cactuBSSN_s', '607', ['0']),
				'lbm'       : benchmark_name('lbm_s',       '619', ['0']),
				'wrf'		: benchmark_name('wrf_s',		'621', ['0']),
				'x264'      : benchmark_name('x264_s',      '625', ['0', '1', '2']),
				'pop2'      : benchmark_name('pop2_s',      '628', ['0']),
				'deepsjeng' : benchmark_name('deepsjeng_s', '631', ['0']),
				'imagick'	: benchmark_name('imagick_s',	'638', ['0']),
				'leela'     : benchmark_name('leela_s',     '641', ['0']),
				'nab'       : benchmark_name('nab_s',       '644', ['0']),
				'exchange2' : benchmark_name('exchange2_s', '648', ['0']),
				'fotonik3d' : benchmark_name('fotonik3d_s', '649', ['0']),
				'roms'      : benchmark_name('roms_s',      '654', ['0']),
				'specrand_f': benchmark_name('specrand_fs', '996', ['0']),
				'specrand_i': benchmark_name('specrand_is', '998', ['0']),
				'xz'		: benchmark_name('xz_s',		'657', ['0', '1']),
				'synthetic' : benchmark_name('synthetic_s', '111', ['0']) # Synthetic benchmark to test transforms
			}

PARSEC_BENCHMARKS = [
						'blackscholes', 'canneal', 'ferret', 'fluidanimate', 'bodytrack',
						'freqmine', 'raytrace', 'streamcluster', 'vips', 'swaptions', 'facesim',
					]

# For normalizing data, most of the times only one counter is needed. But, there are
# a few exceptions. Show all known counters
fixed_counters = ['INST_RETIRED.ANY', 'CPU_CLK_UNHALTED.THREAD', 'CPU_CLK_UNHALTED.REF_TSC']
single_norm_counters = {
	# ARM server counters - map to normalized names
	# Both with and without trailing colons (training_csvs vs final_csvs)
	'instructionspp'                        :   'instructions:u',
	'instructions:pp:'                      :   'instructions:u',
	'cpu-cycles'                            :   'CPI',
	'cpu-cycles:'                           :   'CPI',
	'branch-loads'                          :   'BR_PI',
	'branch-loads:'                         :   'BR_PI',
	'branch-misses'                         :   'BR_MPI',
	'branch-misses:'                        :   'BR_MPI',
	'L1-icache-loads'                       :   'L1_ICACHE_LOADS',
	'L1-icache-loads:'                      :   'L1_ICACHE_LOADS',
	'L1-icache-load-misses'                 :   'L1_ICACHE_MISS',
	'L1-icache-load-misses:'                :   'L1_ICACHE_MISS',
	'L1-dcache-loads'                       :   'L1_DCACHE_LOADS',
	'L1-dcache-loads:'                      :   'L1_DCACHE_LOADS',
	'L1-dcache-load-misses'                 :   'L1_DCACHE_MISS',
	'L1-dcache-load-misses:'                :   'L1_DCACHE_MISS',
	'cache-misses'                          :   'CACHE_MISSES',
	'cache-misses:'                         :   'CACHE_MISSES',
	'cache-references'                      :   'CACHE_REFERENCES',
	'cache-references:'                     :   'CACHE_REFERENCES',
	'iTLB-loads'                            :   'ITLB_LOADS',
	'iTLB-loads:'                           :   'ITLB_LOADS',
	'iTLB-load-misses'                      :   'ITLB_MISS',
	'dTLB-loads'                            :   'DTLB_LOADS',
	'dTLB-loads:'                           :   'DTLB_LOADS',
	'dTLB-load-misses'                      :   'DTLB_MISS',
	'dTLB-load-misses:'                     :   'DTLB_MISS',
	'stalled-cycles-frontend'               :   'STALLED_CYCLES_FRONTEND',
	'stalled-cycles-frontend:'              :   'STALLED_CYCLES_FRONTEND',
	'stalled-cycles-backend'                :   'STALLED_CYCLES_BACKEND',
	'stalled-cycles-backend:'               :   'STALLED_CYCLES_BACKEND',
	'branch-load-misses'                    :   'BRANCH_LOAD_MISSES',
	'branch-load-misses:'                   :   'BRANCH_LOAD_MISSES',
	'bus-cycles'                            :   'BUS_CYCLES',
	'bus-cycles:'                           :   'BUS_CYCLES',
}
single_rate_counters = {
	'L2_RQSTS.DEMAND_DATA_RD_HIT'	:	['L2_RQSTS.ALL_DEMAND_DATA_RD', 'L2_HIT_RATE'],
	'BR_MISP_EXEC.ALL_BRANCHES'		:	['BR_INST_RETIRED.ALL_BRANCHES', 'BR_MISS_RATE'],
	'BR_MISP_RETIRED.ALL_BRANCHES'  :	['BR_INST_RETIRED.ALL_BRANCHES', 'BR_MISS_RATE'],
	'INST_RETIRED.ANY'				:	['CPU_CLK_UNHALTED.THREAD', 'IPC']
}

def get_raw_data(benchmark, dataset, ref=0, folder=DEFAULT_FOLDER):
	# This method reads a CSV file (and it's not compatible with emon format). It
	# returns a pandas Data frame
	# For ARM server data, use the folder directly
	if dataset == 'arm_server':
		set_folder = folder
		# Handle final_csvs structure with frequency subdirectories
		if 'final_csvs' in folder:
			# Extract frequency from benchmark name (e.g., dacapo_biojava_1.5GHz_merged -> 1.5GHz)
			if '1.5GHz' in benchmark:
				freq_folder = '1.5GHz'
				freq = '1.5GHz'
			elif '3.0GHz' in benchmark:
				freq_folder = '3.0GHz'
				freq = '3.0GHz'
			else:
				freq_folder = ''
				filename = benchmark + '.csv'
			
			if freq_folder:
				set_folder = os.path.join(folder, freq_folder)
				# Construct filename: cpu_0_1.5GHz_dacapo_biojava_10000000_merged.csv
				# Remove frequency and _merged from benchmark name
				benchmark_base = benchmark.replace(f'_{freq}_merged', '')
				filename = f'cpu_0_{freq}_{benchmark_base}_10000000_merged.csv'
				path = os.path.join(set_folder, filename)
				if not os.path.exists(path):
					print(f'File not found: {path}')
					return None
		else:
			# Using training_csvs or other folder
			filename = benchmark + '.csv'
			path = os.path.join(set_folder, filename)
	else:
		set_folder = os.path.join(folder, dataset)
	
	# For non-arm_server or non-final_csvs, use the old logic
	if dataset != 'arm_server' or 'final_csvs' not in folder:
		if benchmark in BENCHMARKS:
			bm =  BENCHMARKS[benchmark]
			filename = bm.number + '.' + bm.name + str(ref) + '.csv'
			if str(ref) not in bm.refs:
				print('Introduce a valid reference for %s. Available refs:'%(benchmark))
				print( bm.refs )
				return None
		elif benchmark in PARSEC_BENCHMARKS:
			filename = benchmark + '.csv'
		else:
			filename = benchmark + '.csv'
			if not os.path.exists(os.path.join(set_folder, filename)):
				print('Data for benchmark %s not available'%(benchmark))
				print('Available benchmarks:')
				print(BENCHMARKS.keys())
				print(PARSEC_BENCHMARKS)
				return None
		path = os.path.join(set_folder, filename)
	multicore = False
	with open(path) as f:
		firstline = f.readline().rstrip()
		cpu = firstline.count('CPU')
		coma = firstline.count(',')
		if abs(cpu - coma) < 2:
			multicore = True
	if multicore:
		data = pd.read_csv(path, header=[0,1])
	else:
		data = pd.read_csv(path, header=0)
	return data

def get_processed_data(data):
	if data.columns.nlevels > 1:
		percore = []
		corenames = data.columns.get_level_values(0).unique()
		for core in corenames:
			df = get_processed_data(data[core])
			percore.append(df)
		return pd.concat(percore, axis=1, keys=corenames)
	proc_data = pd.DataFrame()
	missing = []
	instr_counter = ''
	if 'instructions:u' in data.columns:
		instr_counter = 'instructions:u'
	elif 'instructions:pp:' in data.columns:
		instr_counter = 'instructions:pp:'
	elif 'instructionspp' in data.columns:
		instr_counter = 'instructionspp'
	else:
		instr_counter = 'INST_RETIRED.ANY'
	for value in data.columns.values:
		if value in single_norm_counters:
			name = single_norm_counters[value]
			proc_data[name] = data[value] / data[instr_counter]
		elif value not in fixed_counters and value != instr_counter:
			missing.append(value)
	if 'L2_RQSTS.ALL_DEMAND_DATA_RD' in data.columns.values:
		if 'L2_RQSTS.DEMAND_DATA_RD_HIT' in data.columns.values:
			proc_data['L2_MPI'] = (data['L2_RQSTS.ALL_DEMAND_DATA_RD'] - data['L2_RQSTS.DEMAND_DATA_RD_HIT'])/data['INST_RETIRED.ANY']
	for counter in single_rate_counters.keys():
		if counter in data.columns.values:
			pair = single_rate_counters[counter][0]
			if pair in data.columns.values:
				name = single_rate_counters[counter][1]
				proc_data[name] = data[counter] / data[pair]
	if len(missing) > 0:
		print('WARNING: some counters are unknown and not normalized')
		print(missing)
	return proc_data
