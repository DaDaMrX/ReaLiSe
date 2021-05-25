#!/data/anaconda3/bin/python
import sys, os, yaml
from copy import deepcopy
from tqdm import tqdm

def main(yaml_file):
    with open(yaml_file, 'r', encoding='utf-8') as f:
        data = yaml.load(f)

    code_file_dir = data['code']['local_dir']

    run_scripts = []
    template_script = '\n'.join(data['search']['job_template']['command'])

    param_lists = [[]]
    for item in data['search']['params']:
        tmp = deepcopy(param_lists)
        for p in param_lists:
            p.append((item['name'], item['values'][0]))
        for i in range(1, len(item['values'])):
            tmp_b = deepcopy(tmp)
            for p in tmp_b:
                p.append((item['name'], item['values'][i]))
            param_lists += tmp_b
    
    for p in param_lists:
        script = template_script
        for k, v in p:
            script = script.replace('{'+k+'}', str(v))
        run_scripts.append(script)

    print('Total %d Runs.' % len(run_scripts))
    assert len(run_scripts) == len(data['search']['sku'])

    for sku, script in tqdm(zip(data['search']['sku'], run_scripts), desc='submit'):
        dir_path = os.path.join(data['search']['target_dir'], sku)
        run_file = os.path.join(dir_path, 'run_venus.sh')

        #print('cd %s && rm -rf *'%dir_path)
        #print('cp -r %s %s'%(os.path.join(code_file_dir, '*'), dir_path))
        os.system('cd %s && rm -rf * \n'%dir_path)
        
        cmds = '#!/bin/bash\n\n'
        cmds += 'cp -r %s %s \n\n'%(os.path.join(code_file_dir, '*'), dir_path)
        
        with open(run_file, 'w', encoding='utf-8') as writer:
            writer.write(cmds+script)
        os.system('chmod +x %s'%run_file)

if __name__=='__main__':
    main(sys.argv[1])

