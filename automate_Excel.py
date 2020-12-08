
import pandas as pd
import argparse

ap=argparse.ArgumentParser()
ap.add_argument('-f','--folder',type=str,default='',help='path')
args=vars(ap.parse_args())

image_path=args['folder']

df = pd.read_csv(image_path)

#applying conditions
#macro back
mac_tn_b=df.loc[(df['macro_score']>0.5) & (df['nonaadhaar']==True) & (df['type']==1)]
mac_fn_b=df.loc[(df['macro_score']>0.5) & (df['nonaadhaar']==False) & (df['type']==1)]
mac_tp_b=df.loc[(df['macro_score']<0.5) & (df['nonaadhaar']==False) & (df['type']==1)]
mac_fp_b=df.loc[(df['macro_score']<0.5) & (df['nonaadhaar']==True) & (df['type']==1)]
#macro front
mac_tn_f=df.loc[(df['macro_score']>0.5) & (df['nonaadhaar']==True) & (df['type']==0)]
mac_fn_f=df.loc[(df['macro_score']>0.5) & (df['nonaadhaar']==False) & (df['type']==0)]
mac_tp_f=df.loc[(df['macro_score']<0.5) & (df['nonaadhaar']==False) & (df['type']==0)]
mac_fp_f=df.loc[(df['macro_score']<0.5) & (df['nonaadhaar']==True) & (df['type']==0)]

#micro 
mic_tn=df.loc[(df['micro_score']>0.5) &(df['nonaadhaar']==False)&(df['partialaadhaar']==True) & (df['type']==0)]
mic_fn=df.loc[(df['micro_score']>0.5) &(df['nonaadhaar']==False)&(df['partialaadhaar']==False) & (df['type']==0)]
mic_fp=df.loc[(df['micro_score']<0.5) &(df['nonaadhaar']==False)&(df['partialaadhaar']==True) & (df['type']==0)]
mic_tp=df.loc[(df['micro_score']<0.5) &(df['nonaadhaar']==False)&(df['partialaadhaar']==False) & (df['type']==0)]


#redline 
red_tn=df.loc[(df['microredline']>0.5)&(df['nonaadhaar']==False) & (df['redlinecut']==True) & (df['type']==0)]
red_fn=df.loc[(df['microredline']>0.5)&(df['nonaadhaar']==False) & (df['redlinecut']==False) & (df['type']==0)]
red_fp=df.loc[(df['microredline']<0.5)&(df['nonaadhaar']==False) & (df['redlinecut']==True) & (df['type']==0)]
red_tp=df.loc[(df['microredline']<0.5)&(df['nonaadhaar']==False) & (df['redlinecut']==False) & (df['type']==0)]

# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('separator1.xlsx', engine='xlsxwriter')

# seperate dataframe into different worksheets.
#macro back
mac_tn_b.to_excel(writer, sheet_name='mac_tn_b')
mac_fn_b.to_excel(writer, sheet_name='mac_fn_b')
mac_fp_b.to_excel(writer, sheet_name='mac_fp_b')
mac_tp_b.to_excel(writer, sheet_name='mac_tp_b')
#macro front
mac_tn_f.to_excel(writer, sheet_name='mac_tn_f')
mac_fn_f.to_excel(writer, sheet_name='mac_fn_f')
mac_fp_f.to_excel(writer, sheet_name='mac_fp_f')
mac_tp_f.to_excel(writer, sheet_name='mac_tp_f')

#micro
mic_tn.to_excel(writer, sheet_name='mic_tn')
mic_fn.to_excel(writer, sheet_name='mic_fn')
mic_fp.to_excel(writer, sheet_name='mic_fp')
mic_tp.to_excel(writer, sheet_name='mic_tp')


#redline
red_tn.to_excel(writer, sheet_name='red_tn')
red_fn.to_excel(writer, sheet_name='red_fn')
red_fp.to_excel(writer, sheet_name='red_fp')
red_tp.to_excel(writer, sheet_name='red_tp')


# Close the Pandas Excel writer and output the Excel file.
writer.save()
