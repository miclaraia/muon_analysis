
# pylint: disable=C0103
tool_name = 'Tool name'
task_A = 'T0'
task_B = ['T1', 'T2']
launch_date = '2019-03-10'

time_format = '%Y-%m-%d %H:%M:%S %Z'
image_groups = [10, 11, 12, 13]
image_dir = 'path/to/images'

task_map = {
    'all_muons': ['All are Muons'],
    'most_muons': ['Majority are Muons', 'No clear majority'],
    'most_nonmuons': ['Majority are **not** Muons'],
    'no_muons': ['**None** are Muons']
}
