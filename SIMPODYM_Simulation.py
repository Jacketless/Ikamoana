from argparse import ArgumentParser
from SEAPODYM_PARCELS import *

if __name__ == "__main__":
    p = ArgumentParser(description="""
                       Arguments for parameterising a series of SIMPODYM runs""")
    p.add_argument('-f', '--parameter_file', type=str, default='SIMPODYM_Parameters.txt',
                   help='file path of parameter text file')
    args = p.parse_args()

    list = ['output', 'mode', 'reps',
            'individuals', 'timestep', 'time ', 'start_age',
            'Ufile', 'Vfile', 'Hfile', 'dHdx_file', 'dHdy_file',
            'Uvar_name', 'Vvar_name', 'Hvar_name',
            'londim_name', 'latdim_name', 'timedim_name',
            'starting_density_file', 'density_var_name', 'calculate_density',
            'write_density', 'write_grid',
            'Kfile', 'dKdx_file', 'dKdy_file', 'diffusion_boost', 'taxis_dampener']

    plist = {}
    for p in list:
        plist.update({p: None})

    #f = open(args.parameter_file,'r')
    for line in open(args.parameter_file, 'r'):
        for p in plist:
            if p in line:
                plist[p] = line.split()[-1]

    # Set defaults for key params of import
    if plist['mode'] is None:
        plist['mode'] = 'jit'
    if plist['reps'] is None:
        plist['reps'] = 1
    if plist['individuals'] is None:
        plist['individuals'] = '100'
    if plist['time '] is None:
        plist['time'] = 30
    if plist['timestep'] is None:
        plist['timestep'] = 172800
    if plist['start_age'] is None:
        plist['start_age'] = 4
    if plist['write_grid'] is None:
        plist['write_grid'] = False
    if plist['diffusion_boost'] is None:
        plist['diffusion_boost'] = 0
    if plist['taxis_dampener'] is None:
        plist['taxis_dampener'] = 0

    # Check that key params exist
    key_params = ['Ufile', 'Vfile', 'Hfile', 'Uvar_name', 'Vvar_name', 'Hvar_name',
                  'londim_name', 'latdim_name', 'timedim_name', 'output']
    for p in key_params:
        if plist[p] is None:
            print("No defined %s!" % p)
            sys.exit()

    print("Parameters for simulations")
    for p in plist.keys():
        print("%s = %s" % (p, plist[p]))

    print('Beginning Simulations')
    for rep in plist['reps']:
        if plist['reps'] > 1:
            outputname = plist['output']+str(rep)
        else:
            outputname = plist['output']
        SIMPODYM(forcingU=plist['Ufile'], forcingV=plist['Vfile'], forcingH=plist['Hfile'],
                 startD=plist['starting_density_file'], Dname=plist['density_var_name'], output_density=plist['write_density'],
                 Uname=plist['Uvar_name'], Vname=plist['Vvar_name'], Hname=plist['Hvar_name'],
                 dimLon=plist['londim_name'], dimLat=plist['latdim_name'], dimTime=plist['timedim_name'],
                 Kfile=plist['Kfile'], dK_dxfile=plist['dKdx_file'], dK_dyfile=plist['dKdy_file'], diffusion_boost=plist['diffusion_boost'],
                 dH_dxfile=plist['dHdx_file'], dH_dyfile=plist['dHdy_file'],
                 individuals=int(plist['individuals']), timestep=float(plist['timestep']), time=int(float(plist['time '])),
                 start_age=float(plist['start_age']),
                 output_file=outputname, mode=plist['mode'], write_grid=plist['write_grid'])
