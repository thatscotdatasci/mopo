############################################################################
# NOTE: This script will modify log files - keeping commented out to prevent
# accidental execution.
############################################################################

######################################################################################
# The purpose of this script is to trim the training-logs down to only the first 123
# lines. The files can take a considerable amount of space, but are mostly not used.
#
# The first 123 lines are kept as all the information that might be used is contained
# within these (specifically, the choice of elite models has been made and declared by
# this point).
######################################################################################

# import os
# from glob import glob

# FULL_LOG_PATH = os.path.expanduser('~/rds/rds-dsk-lab-eWkDxBhxBrQ/dimorl/code/dogo_results/mopo/ray_mopo/full_logs')

# def main():
#     for path in glob(os.path.expanduser('~/rds/rds-dsk-lab-eWkDxBhxBrQ/dimorl/code/dogo_results/mopo/ray_mopo/halfcheetah/*/seed*/train-log.*')):
#         trim_filepath = os.path.join(FULL_LOG_PATH, *path.split('/')[-4:])
#         trim_dir = os.path.dirname(trim_filepath)
#         if not os.path.isfile(trim_filepath):
#             os.makedirs(trim_dir, exist_ok=True)

#             try:
#                 with open(path, 'r') as f:
#                     trimmed_log = [next(f) for _ in range(123)]
#             except StopIteration:
#                 continue

#             os.rename(path, trim_filepath)

#             with open(path, 'w') as f:
#                 f.writelines(trimmed_log)
#         else:
#             print(f'File already exists: {trim_filepath}')
#             continue

# if __name__ == "__main__":
#     main()
