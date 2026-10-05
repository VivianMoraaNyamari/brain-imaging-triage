from __future__ import annotations

from config import PROJECT_ROOT
from src.data_utils import summarize_dataset


def main():
    print('Project root:', PROJECT_ROOT)
    print('Dataset summary:')
    dataset_summary = summarize_dataset()
    if not dataset_summary:
        print('No dataset files found yet. Create the train/val/test folders and populate them.')
    else:
        for item in dataset_summary:
            print(item)
    print('\nPipeline scaffold is ready.')


if __name__ == '__main__':
    main()
