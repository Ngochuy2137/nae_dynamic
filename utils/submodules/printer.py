class Printer:
    def __init__(self):
        pass

    def print_green(self, text, background=False, enable=True):
        if not enable:
            return
        if background:
            print('\033[42m' + text + '\033[0m')
        else:
            print('\033[92m' + text + '\033[0m')
    
    def print_red(self, text, background=False, enable=True):
        if not enable:
            return
        if background:
            print('\033[41m' + text + '\033[0m')
        else:
            print('\033[91m' + text + '\033[0m')

    def print_yellow(self, text, background=False, enable=True):
        if not enable:
            return
        if background:
            print('\033[43m' + text + '\033[0m')
        else:
            print('\033[93m' + text + '\033[0m')

    def print_blue(self, text, background=False, enable=True):
        if not enable:
            return
        if background:
            print('\033[44m' + text + '\033[0m')
        else:
            print('\033[94m' + text + '\033[0m')

    def print_purple(self, text, background=False, enable=True):
        if not enable:
            return
        if background:
            print('\033[45m' + text + '\033[0m')
        else:
            print('\033[95m' + text + '\033[0m')
