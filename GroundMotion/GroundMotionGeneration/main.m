clear;
clc;

path = "C:\Users\v196m\Desktop\master_project\Masterarbeit\GroundMotion\RecordData\GM_2012_11_12_11_15_04\GM_INSH_2012_11_12_11_15_04\a_1_INSH_2012_11_12_11_15_04.txt";
a = fns_import_time_data(path,'txt',false);

b = fns_fft_data(a,100);

