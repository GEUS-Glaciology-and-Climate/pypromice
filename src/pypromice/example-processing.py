
from aws import AWS_bav
import matplotlib.pyplot as plt
        
path_to_l0 = 'C:/Users/bav/OneDrive - Geological survey of Denmark and Greenland/Code/PROMICE/aws-l0/'

test = AWS_bav(config_file=path_to_l0 + 'tx/config/KPC_Uv3.toml', inpath=path_to_l0 + 'tx', outpath='./test')
L0 = test.L0.copy()

plt.close('all')
plt.figure()
L0[0].t_u.plot()

plt.figure()
test.L1[0].t_u.plot()
