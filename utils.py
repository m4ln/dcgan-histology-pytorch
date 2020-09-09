import errno
import os
import shutil
import sys
from pathlib import Path

# check os for sds path
def check_os():
    if sys.platform == "linux":
        path1 = '/home/marlen/sds_hd/sd18a006/'
        path2 = '/home/mr38/sds_hd/sd18a006/'
        if Path(path1).exists():
            return path1
        elif Path(path2).exists():
            return path2
        else:
            print('error: sds path cannot be defined! Abort')
            return 1
    elif sys.platform == "win32":
        path = '//lsdf02.urz.uni-heidelberg.de/sd18A006/'
        if Path(path).exists():
            return path
        else:
            print('error: sds path cannot be defined! Abort')
            return 1
    else:
        print('error: sds path cannot be defined! Abort')
        return 1

def to_np(var):
    """Exports torch.Tensor to Numpy array.
    """
    return var.detach().cpu().numpy()


def create_folder(folder_path):
    """Create a folder if it does not exist.
    """
    try:
        os.makedirs(folder_path)
    except OSError as _e:
        if _e.errno != errno.EEXIST:
            raise


def clear_folder(folder_path):
    """Clear all contents recursively if the folder exists.
    Create the folder if it has been accidently deleted.
    """
    create_folder(folder_path)
    for the_file in os.listdir(folder_path):
        _file_path = os.path.join(folder_path, the_file)
        try:
            if os.path.isfile(_file_path):
                os.unlink(_file_path)
            elif os.path.isdir(_file_path):
                shutil.rmtree(_file_path)
        except OSError as _e:
            print(_e)


class StdOut(object):
    """Redirect stdout to file, and print to console as well.
    """
    def __init__(self, output_file):
        self.terminal = sys.stdout
        self.log = open(output_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'
