############################################################################
# NOTE: This script will modify log files - keeping commented out to prevent
# accidental execution.
############################################################################

# import os
# from glob import glob

# FULL_LOG_PATH = '/home/ajc348/rds/hpc-work/dogo_results/mopo/ray_mopo/full_logs'

# def main():
#     for path in glob('/home/ajc348/rds/hpc-work/dogo_results/mopo/ray_mopo/halfcheetah/*/seed*/train-log.*'):
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
