def combine(file, save_file):
    with open(file, 'r') as f:
        data = [line.strip('\n').strip() for line in f.readlines()]
    src = []
    cnt = 0
    tmp = []
    for line in data:
        cnt += 1
        tmp.append(line)
        if cnt == 5:
            cnt = 0
            src.append(' '.join(tmp))
            tmp = [] 

    with open(save_file, 'w') as f:
        for line in src:
            f.write(line)
            f.write('\n')
file = '/home/linzhe/self_key2/data/output/result.txt'
save_file = file + '.combine'
combine(file, save_file)