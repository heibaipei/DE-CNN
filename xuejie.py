'''
Created on 2018年3月22日
   
@author: Jack
'''
   
import os
import numpy
   
   
def get_student_data_paths(students_data_dir='data'):
    '''
    @param students_data_dir:the directory path where keep the student_datas
    @return student_data_paths
    '''
    student_data_paths = os.listdir(students_data_dir)
    return student_data_paths
   
   
def get_a_student_eegs(student_data_path):
    '''
    @param mat_path:
    @return student_name:
    @return eegs:
    '''
    import scipy.io
       
    student_name = os.path.basename(student_data_path).replace('.mat', '')
       
    a_student_data = scipy.io.loadmat(student_data_path)
    eeg_keys = [i for i in a_student_data.keys() if i[0] != '_']

    eegs = []
    for i in eeg_keys:
        eegs.append(a_student_data[i])
           
    return student_name, eegs
   
   
# def split_an_eeg(eeg, second_n=5, a_second_sample_n=200):
#     '''
#     @param eeg:eeg data
#     @param second_n:split a egg by this many seconds
#     @param a_second_sample_n:how many sample can be captch in a second
#     @return splited_eegs
#     '''
#
#     step_length = second_n * a_second_sample_n
#     _, n = eeg.shape
#     part_n = n // step_length
#
#     # 无法平均切割，则舍弃后面的数据
#     eeg = eeg[:, :part_n * step_length]
#     splited_eegs = numpy.split(eeg, part_n, axis=1)
#
#     return splited_eegs
   
   
def save_splited_eeg(splited_eeg, save_dir='processed_EEG', save_name='you_konw.pkl'):
    '''
    @param splited_eeg:egg that has already been splited
    @param save_dir:the dir where to save the splited eeg
    @param save_name: you know 
    @return nothing
    '''
    import pickle
    import scipy.io as scio  

    save_path = os.path.join(save_dir, save_name)
    try:
#         with open(save_path, 'wb') as fp:
        scio.savemat(save_path, splited_eeg)  
#             pickle.dump(splited_eeg, fp, protocol=4)
    except:
        try:
            os.mkdir(save_dir)
        except:
            pass
#         with open(save_path, 'wb') as fp:
        scio.savemat(save_path, {'key':splited_eeg})  
#             pickle.dump(splited_eeg, fp, protocol=4)

def get_label(record):
    l = len(record)
    numbers = []
    i = 0
    while i < l:
        num = ''
        symbol = record[i]
        while '0' <= symbol <= '9':  # symbol.isdigit()
            num += symbol
            i += 1
            if i < l:
                symbol = record[i]
            else:
                break
        i += 1
        # if num != '':
        #     numbers.append(int(num))
        if num !='':
            Digit = int(num)
    return Digit
   
   
def generate_train_test_data(students_data_dir='data', second_n=3):
    '''
    @param 
    @return nothing
    '''
    student_data_paths = get_student_data_paths(students_data_dir='data')
       
    for student_data_path in student_data_paths:
           
        student_name, eegs = get_a_student_eegs(student_data_path=os.path.join(students_data_dir, student_data_path))
        
        for i, eeg in zip(range(len(eegs)), eegs):
            
            try:
                splited_eegs = split_an_eeg(eeg=eeg, second_n=second_n, a_second_sample_n=200)
                
                if i < 10:
                    save_splited_eeg(splited_eegs, save_dir=os.path.join('processed_EEG', 'train'), save_name=student_name + '_' + str(i))
                else:
                    save_splited_eeg(splited_eegs, save_dir=os.path.join('processed_EEG', 'test'), save_name=student_name + '_' + str(i))
                
                # output so we can see where is wrong
                print(numpy.array(splited_eegs).shape)
    #             print(splited_eegs)
                print('==================================================================')
            except:
                pass
   
   
# if __name__ == '__main__':
#     generate_train_test_data(second_n=3)