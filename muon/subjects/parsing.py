import re
import os
import h5py
from tqdm import tqdm

from muon.subjects import Subject


class HDFParseQi:


#     def __init__(self, data_file):
#         self.num = 0
#         self.output = data_file
# 
#         self._file = None
# 
#     # def to_subjects(cls, data_file):
#         # if not os.path.isfile(data_file):
#             # raise IOError('Data file doesn\'t exist!')
# 
#         # subject_data = Subject_Data(data_file)
#         # subjects = {}
#         # for subject, evt, charge in subject_data:
#             # if subject is not None:
#                 # charge = charge[:-1]
#                 # # TODO this isn't right anymore
#                 # # TODO make sure to add run,evt,tel to subject metadata
#                 # s = Subject(subject, evt, charge)
#                 # subjects[subject] = s
#             # else:
#                 # raise Exception('Subject id was None ....')
# 
#         # return cls(subjects)
# 
    patterns = {
        'run': re.compile('run([0-9]+)'),
        'evt': re.compile('evt([0-9]+)'),
        'tel': re.compile('tel([0-9]+)'),
    }
# 
#     @property
#     def file(self):
#         if self._file is None:
#             self._file = self.load()
#         return self._file
# 
#     def load(self):
#         if os.path.isfile(self.output):
#             file = h5py.File(self.output, 'r+')
#             self.num = file['stats'].attrs['num']
#         else:
#             file = h5py.File(self.output, 'w')
#             stats = file.create_group('stats')
#             stats.attrs['num'] = self.num
#             file.create_group('data')
# 
#         return file
# 
#     def close(self):
#         self.file.close()
# 
#     def __iter__(self):
#         for run in self.file['data']:
#             run = self.file['data'][run]
#             for event in run:
#                 event = run[event]
#                 _event = (
#                     event.attrs['run'],
#                     event.attrs['evt'],
#                     event.attrs['tel']
#                 )
#                 subject = event.attrs['subject']
#                 charge = event['charge']
# 
#                 yield (subject, _event, charge)
# 
#     def load_raw(self, args):
#         for run, event, charge in self.raw_files(args):
#             self.add(run, event, charge)
# 
#         self.close()
# 
#     def add(self, run, event, charge):
#         run, evt, tel = self.parse_event(run, event)
#         _run = str(run)
#         _evt = str(evt)
# 
#         data = self.file['data']
#         if _run not in data:
#             data.create_group(_run)
# 
#         if _evt not in data[_run]:
#             e = data[_run].create_group(_evt)
#             e.attrs.update({'tel': tel, 'run': run, 'evt': evt})
#             e.attrs['subject'] = self.num
#             e.create_dataset('charge', charge.shape,
#                              data=charge, compression='gzip')
# 
#             self.num += 1
#             self.file['stats'].attrs['num'] = self.num

    ##########################################################################
    ###   Loading Original Data   ############################################
    ##########################################################################

    @classmethod
    def parse_event(cls, run, evt):
        def parse(regex, string):
            s = regex.search(string)
            return int(s.group(1))

        run = parse(cls.patterns['run'], run)
        event = parse(cls.patterns['evt'], evt)
        tel = parse(cls.patterns['tel'], evt)

        return (run, event, tel)

    @classmethod
    def raw_files(cls, args):
        print('loading files from %s' % str(args))
        paths = []
        for path in args:
            print(path)
            if os.path.isdir(path):
                for fname in os.listdir(path):
                    print(fname)
                    if os.path.splitext(fname)[1] == '.hdf5':
                        if path not in paths:
                            paths.append(os.path.join(path, fname))

            elif os.path.splitext(path)[1] == '.hdf5':
                if path not in paths:
                    paths.append(path)

        print('loading paths %s' % paths)
        for fname in tqdm(paths):
            # for fname in paths:
            for item in cls.raw_file(fname):
                yield item

    @classmethod
    def raw_file(cls, fname):
        print('Loading subjects from %s' % fname)
        with h5py.File(fname) as file:
            for run in file:
                for event in tqdm(file[run]):
                    # for event in file[run]:
                    if event == 'summary':
                        continue
                    try:
                        charge = file[run][event]['charge']
                    except KeyError:
                        print(run, event)
                        raise

                    id_ = cls.parse_event(run, event)
                    yield(id_, charge)

