import sys
from math import inf
from configparser import ConfigParser, ExtendedInterpolation

def readConfig(configPath, globalVars = globals(), localVars = locals()):
	'''
	Wrapper for the Python config parser to read an ini config file and return
	a dictionary of typed parameters. For documentation of Python configparser
	and ini use, see https://docs.python.org/3.8/library/configparser.html

	Expects a valid filepath to the config file as input
	'''

	params = dict()
	config = ConfigParser(inline_comment_prefixes=('#'),interpolation=ExtendedInterpolation())
	config.optionxform = lambda option: option
	config.read(configPath)
	for section in config:
		params[section] = dict()
		for key in config[section]:
			params[section][key] = interpolate(config[section],key, globalVars, localVars)
	return params

def interpolate(config, key, globalVars, localVars):
	'''
	Attempts to interpolate parameters to more useful data types based on 
	semi-intelligent methods provided by configparser. 
	
	Type precedence:
	int
	float
	boolean
	expression
	string

	Ambiguous numerical parameters default to ints. Floats are identified if
	the float and int values differ (so 1.0 would be cast to an int). 0 and 1
	are interpreted as ints instead of boolean values under the assumption that
	this doesn't impact logical operations on the values. If boolean, float,
	and	int types fail, the parameter is assumed to be a string type. An
	attempt is made to evaluate the string as a Python expression. If 
	successful, the expresison result is returned. Otherwise, the parameter is
	assumed to actually be a string.

	Floats accept scientific notation such as 1E3 for 1000

	Booleans accept a range of (case-insensitive) values: 
	True/False
	yes/no
	on/off
	1/0 (though this one is converted to int as documented above)
	'''
	try:
		floatNum = config.getfloat(key)
		try:
			intNum = config.getint(key)
			if floatNum == intNum:
				return intNum
			else:
				return floatNum
		except:
			return floatNum
	except:
		pass
	
	try:
		return config.getboolean(key)
	except:
		pass
	
	try:
		return eval(config.get(key), globalVars, localVars) # evaluate expressions
	except:
		pass
	
	return config.get(key) # returns a string


def checkValue(dir, key, value, checker_set):
	if checker_set[0]:
		if not isinstance(value, checker_set[0]):
			print(dir + ": The type of " + key + " is incorrect.")
			sys.exit()
	if checker_set[1]:
		if value <= checker_set[1]:
			print(dir + ": The value of "+ key +' should be greater than ' + str(checker_set[1])+'.')
			sys.exit()
	if checker_set[2]:
		if value >= checker_set[2]:
			print(dir + ": The value of "+ key +' should be less than ' + str(checker_set[2])+'.')
			sys.exit()

	if checker_set[3]:
		if value not in checker_set[3]:
			print(dir + ": The value of " + key + ' is not in the options: ' + str(checker_set[3]) + '.')
			sys.exit()


def checkConfig(config):
	print('configs:',config)
	# (type,min,max,options)
	REQUIRED_CONFIG_CHECK = {
		'DEFAULT': {},
		'fitness_kwargs': {
			'pooling': (str, None, None, ['genetic','ea']),
			'model': (str, None, None, ['vgg11_bn','vgg13_bn','vgg16_bn','vgg19_bn']),
			'epoch': (int, 0, inf, None),
			'dataset': (str, None, None, ['cifar100','cifar10','mnist','svhn']),
			'fitness_metric': (str, None, None, ['best_acc','gen_gap','best_op','ea_op','best_acc&op','best_acc&op&gen'])
		},

		'EA_configs': {
			'seed': (int, -inf, inf, None),
			'run': (int, 0, inf, None),
			'number_evaluations': (int, 0, inf, None),
			'mu': (int, 0, inf, None),
			'num_children': (int, 0, inf, None),
			'mutation_rate': (float, 0, 1, None),
			'individual_class': (None,None,None,None),
			'parent_selection': (None,None,None,None),
			'survival_selection': (None,None,None,None)},
		'initialization_kwargs':
				{	'internal_nodes': (set,None,None,None),
					'leaf_nodes': (set, None, None, None),
					 'seq_length': (int, 0, inf, None),
					'depth_limit': (int, 0, inf, None)},
		'parent_selection_kwargs':
				{'k': (int, 0, inf, None)},
		'recombination_kwargs':
				{},
		'mutation_kwargs':
				{},
		'survival_selection_kwargs': {}
	}

	REQUIRED_CONFIG_KEYS = REQUIRED_CONFIG_CHECK.keys()

	check_dir = '[configs]'
	for required_key in REQUIRED_CONFIG_KEYS:
		if required_key not in config.keys():
			print(check_dir+required_key+" is missing.")
			sys.exit()
		for sub_require_key in 	REQUIRED_CONFIG_CHECK[required_key].keys():
			sub_dir = check_dir+">"+'['+str(required_key)+']'
			if sub_require_key not in config[required_key].keys():
				print(sub_dir + ": " + sub_require_key + " is missing.")
				sys.exit()
			checkValue(sub_dir, sub_require_key, config[required_key][sub_require_key], REQUIRED_CONFIG_CHECK[required_key][sub_require_key])

	# Additional Check

	if config['EA_configs']['mu']<config['parent_selection_kwargs']['k']:
		print('The population mu in EA_configs must be larger than k in parent_selection_kwargs.')
		sys.exit()

	if isinstance(config['EA_configs']['individual_class'],str):
		print('The individual_class should not be string. Add \"from rtreeGenotype import *\" or \"from treeGenotype import *\"to your main.py.')
		sys.exit()

	if isinstance(config['EA_configs']['parent_selection'],str):
		print('The parent_selection should not be string. Add \"from selection import *\" to your main.py.')
		sys.exit()

	if isinstance(config['EA_configs']['survival_selection'],str):
		print('The survival_selection should not be string. Add \"from selection import *\" to your main.py.')
		sys.exit()




if __name__ == '__main__':
	print(readConfig("./samples/example.cfg"))