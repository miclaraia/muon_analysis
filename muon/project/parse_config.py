
class Config:
    tool_name = 'Tool name'
    task_A = 'T0'
    task_B = ['T1', 'T3']
    launch_date = '2000-01-31'

    time_format = '%Y-%m-%d %H:%M:%S %Z'
    image_groups = [1,2,3]
    image_dir = 'path/to/images'

    task_map = {
        'all_muons': ['All Muons'],
        'most_muons': ['Majority are Muons', 'No clear majority'],
        'most_nonmuons': ['Minority are Muons'],
        'no_muons': ['No Muons']
    }

    subject_path = '/path/to/subjects'

