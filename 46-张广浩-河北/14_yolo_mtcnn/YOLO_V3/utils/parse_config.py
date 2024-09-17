

def parse_model_config(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    #读取绝对路径
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines] # get rid of fringe whitespaces
    module_defs = []
    for line in lines:
        #【】内为一个bolck
        if line.startswith('['): # This marks the start of a new block
            module_defs.append({})
            # type 为【中间值】
            module_defs[-1]['type'] = line[1:-1].rstrip()
            #如果是卷积操作，batch_normalize默认为0，不加批归一化
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs

def parse_data_config(path):
    """Parses the data configuration file"""
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options

if __name__ == '__main__':
    model_def = "D:\python\YOLO\YOLO_V3\config\yolov3.cfg"
    module_defs = parse_model_config(model_def)
    #print(module_defs)
    for item in module_defs:
        for info in item:
            message = info.title() + ":" + str(item[info]).title()
            print(message)
        print("-"*50)
