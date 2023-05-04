from utils.freq_imports import *


def create_dir(path_to_dir, remove_existing):
    """Creates a new directory at path."""
    path_to_dir = Path(path_to_dir)

    if remove_existing and path_to_dir.exists():
        shutil.rmtree(path_to_dir)
        logging.info('Creating new directory at %s', str(path_to_dir))
    path_to_dir.mkdir(parents=True, exist_ok=True)