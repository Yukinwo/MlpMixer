import os

def get_dir_pathes():
    root_dir = 'H:/stem cell/MLP-Mixer/dataset'
    all_dirs = os.listdir(root_dir)
    train_dir_list = [dir for dir in all_dirs if dir[0] == 'a' or dir[0] == 'o' or dir[:2] == 'n_']
    test_dir_list = [dir for dir in all_dirs if dir[0] != 'a' and dir[0] != 'o' and dir[:2] != 'n_']
    return train_dir_list, test_dir_list

def get_file_pathes(train_dir_list, test_dir_list):
    root_dir = 'H:/stem cell/MLP-Mixer/dataset'
    train_files = []
    a_test_files = []
    o_test_files = []
    n_test_files= []
    nt3_test_files = []
    nt4_test_files = []
    mt_test_files = []
    ngf_test_files = []
    cntf_test_files = []
    ln_test_files = []
    for train_dir in train_dir_list:
        if train_dir[0] == 'a':
            for file in os.listdir(os.path.join(root_dir, train_dir)):
                a_test_files.append(os.path.join(train_dir, file))
        elif train_dir[0] == 'o':
            for file in os.listdir(os.path.join(root_dir, train_dir)):
                o_test_files.append(os.path.join(train_dir, file))
        elif train_dir[:2] == 'n_':
            for file in os.listdir(os.path.join(root_dir, train_dir)):
                n_test_files.append(os.path.join(train_dir, file))
    for test_dir in test_dir_list:
        if test_dir[0] == 'c':
            for file in os.listdir(os.path.join(root_dir, test_dir)):
                cntf_test_files.append(os.path.join(test_dir, file))
        elif test_dir[0] == 'l':
            for file in os.listdir(os.path.join(root_dir, test_dir)):
                ln_test_files.append(os.path.join(test_dir, file))
        elif test_dir[:3] == 'ngf':
            for file in os.listdir(os.path.join(root_dir, test_dir)):
                ngf_test_files.append(os.path.join(test_dir, file))
        elif test_dir[:2] == 'mt':
            for file in os.listdir(os.path.join(root_dir, test_dir)):
                mt_test_files.append(os.path.join(test_dir, file))
        elif test_dir[:3] == 'nt3':
            for file in os.listdir(os.path.join(root_dir, test_dir)):
                nt3_test_files.append(os.path.join(test_dir, file))
        elif test_dir[:3] == 'nt4':
            for file in os.listdir(os.path.join(root_dir, test_dir)):
                nt4_test_files.append(os.path.join(test_dir, file))
    for index, file in enumerate(a_test_files):
        if index < int(len(a_test_files) * 0.8):
            train_files.append(file)
    for index, file in enumerate(o_test_files):
        if index < int(len(o_test_files) * 0.8):
            train_files.append(file)
    for index, file in enumerate(n_test_files):
        if index < int(len(n_test_files) * 0.8):
            train_files.append(file)
    a_test_files = list(set(a_test_files) - set(train_files))
    o_test_files = list(set(o_test_files) - set(train_files))
    n_test_files = list(set(n_test_files) - set(train_files))
    test_files = {
        'a': a_test_files,
        'o': o_test_files,
        'n': n_test_files,
        'nt3': nt3_test_files,
        'nt4': nt4_test_files,
        'ln': ln_test_files,
        'cntf': cntf_test_files,
        'ngf': ngf_test_files,
        'mt': mt_test_files
    }
    return train_files, test_files

if __name__ == '__main__':
    pass