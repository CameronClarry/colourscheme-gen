# Some programs do not allow access to X resources colours from their config files.
# For these cases, this script fills a template config with the correct colours.
# The config files are specified in templates.txt in the root directory of the repository.
import subprocess
import os
import re

# Query xrdb
result = subprocess.run(['xrdb', '-query'], stdout=subprocess.PIPE)
lines = result.stdout.decode('utf-8').split('\n')

# Read the output to get the settings
settings_re = re.compile(r'(.*)\.(.*):\s*([^\s]+)')
settings = {}
for line in lines:
    match = re.match(settings_re, line)
    if match:
        program = match.group(1)
        field = match.group(2)
        value = match.group(3)
        if not program in settings:
            settings[program] = {}

        settings[program][field] = value

# Read the list of templates to fill out
programs = {}
template_list_path = os.path.join(os.path.dirname(__file__), 'templates.txt')
try:
    with open(template_list_path, 'r') as f:
        for line in f.readlines():
            parts = line.split(':')
            programs[parts[0]] = parts[1].strip()
except FileNotFoundError:
    with open(template_list_path, 'w'):
        print('No template list found, made an empty file')

# Make the substitutions in each program's template file, then save the config file
settings_re = re.compile(r'\${(.*)}')
for program in programs:
    config_path = programs[program]
    template_path = '%s.template'%config_path
    if not program in settings:
        print('No settings found for %s, skipping'%program)
        continue

    if not os.path.isfile(template_path):
        print('Could not find the template for %s with path %s, skipping'%(program, template_path))
        continue

    lines_out = []
    with open(template_path, 'r') as tfile:
        lines_in = tfile.readlines()
        for line in lines_in:
            match = re.search(settings_re, line)
            if match:
                field = match.group(1)
                if field in settings[program]:
                    line = re.sub(settings_re, settings[program][field], line)

            lines_out.append(line)

    with open(config_path, 'w') as cfile:
        cfile.writelines(lines_out)

