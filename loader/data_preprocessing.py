import csv
from datetime import datetime
import numpy as np
np.set_printoptions(threshold=np.inf)


def read_test_csv(test_csv=None):
    if test_csv:
        csv_file = open('./data/CoolingLoad7days.csv', 'r', encoding='utf-8', newline='')
    else:
        csv_file = open('./data/CoolingLoad7days.csv', 'r', encoding='utf-8', newline='')
    csv_reader_lines = csv.reader(csv_file)
    a = []
    number = 0
    for one_line in csv_reader_lines:
        a.append(one_line)
        number = number + 1
    s = np.array(a)
    return s


def read_train_csv():
    csv_file = open('./data/imputed_18months.csv', 'r', encoding='utf-8', newline='')
    csv_reader_lines = csv.reader(csv_file)
    a = []
    number = 0
    for one_line in csv_reader_lines:
        a.append(one_line)
        number = number + 1
    s = np.array(a)
    print(s.shape)
    
    result_time = []
    for each in s[1:s.shape[0], :]:
        date = datetime.strptime(each[0], '%Y/%m/%d %H:%M')
        x = date.timetuple().tm_yday
        y = date.weekday()
        month_day_year = str(date.strftime('%Y/%m/%d'))
        hour_of_day = date.timetuple().tm_hour
        min_of_hour = date.timetuple().tm_min
        each = np.insert(each, 0, values=month_day_year, axis=0)
        each = np.insert(each, 1, values=hour_of_day, axis=0)
        each = np.insert(each, 2, values=min_of_hour, axis=0)
        each = np.insert(each, 8, values=0, axis=0)
        each = np.insert(each, 9, values=y, axis=0)
        each = np.insert(each, 10, values=x, axis=0)
        result_time.append(each.tolist())
    result_time = np.array(result_time)
    return result_time


def imputed_zero_average(s):
    for i in range(1, s.shape[0]):
        for j in range(s.shape[1]):
            if s[i][j] == '':
                temp = 0
                number = 0
                if s[i - 1][j] != '':
                    temp = float(s[i - 1][j]) + temp
                    number = number + 1
                if s[i + 1] != '':
                    temp = float(s[i + 1][j]) + temp
                    number = number + 1
                if number == 0:
                    s[i][j] = '0'
                else:
                    s[i][j] = str(temp / number)
    return s


def imputed_below_zero(s):
    for i in range(1, s.shape[0]):
        for j in range(1, s.shape[1]):
            if s[i][j] = '':
                continue
            else:
                temp = float(s[i][j])
                if temp < 0:
                    s[i][j] = ''
    return s


def insert_times_and_reshape(result):
    result = result[1:result.shape[0], :]
    result_time = []
    for each in result:
        date = datetime.strptime(each[0], '%Y-%m-%d %H:%M:%S')
        x = date.timetuple().tm_yday
        y = date.weekday()
        month_day_year = str(date.strftime('%Y/%m/%d'))
        hour_of_day = date.timetuple().tm_hour
        min_of_hour = date.timetuple().tm_min
        each = np.insert(each, 0, values=month_day_year, axis=0)
        each = np.insert(each, 1, values=hour_of_day, axis=0)
        each = np.insert(each, 2, values=min_of_hour, axis=0)
        each = np.insert(each, 8, values=0, axis=0)
        each = np.insert(each, 9, values=y, axis=0)
        each = np.insert(each, 10, values=x, axis=0)
        result_time.append(each.tolist())
    everyday_a = []
    final_result = []
    result_time = np.array(result_time)
    i = 0
    while i < result_time.shape[0]:
        j = i
        while j < i + 96:
            temp = result_time[j].tolist()
            everyday_a.append(temp)
            j = j + 1
        final_result.append(everyday_a)
        everyday_a = []
        i = i + 96
    final_result = np.array(final_result)
    return final_result


def insert_prototype(a,prototype18=None):
    if prototype18:
        pass
    else:
        prototype18 = np.load("./data/prototype18.npy")
    for i in range(a.shape[0]):
        a[i, :, 8] = prototype18[int(a[i, 0, 9])].repeat(4)
        if (('2020' in a[i, 0, 0]) and (
                int(a[i, 0, -1]) in [1, 25, 27, 28, 95, 101, 102, 104, 121, 122, 177, 183, 275, 276, 360, 361])) or (
                ('2021' in a[i, 0, 0]) and (
                int(a[i, 0, -1]) in [1, 43, 44, 46, 92, 93, 95, 96, 121, 122, 139, 165, 182, 265, 274, 287, 359, 361])):
            a[i, :, 8] = prototype18[6].repeat(4)
    return a


def test_data_preprocessing(prototype):
    s = read_test_csv()
    # result = imputed_zero_average(s)
    result = imputed_below_zero(s)
    final = insert_times_and_reshape(result)
    a = insert_prototype(final,prototype)
    np.save("test_model.npy", a)
    x = np.load("test_model.npy")
    print(x.shape)
    return a

def get_day_index(years,dayofweek,dates):
    day_list = [[] for i in range(7)]
    for day in [0,1,2,3,4,5,6]:
        for i,(year,dow,date) in enumerate(zip(years,dayofweek,dates)):
            holiday = (int(year[0][:4]) == 2020 and int(date[0]) in [1,25,27,28,95,101,102,104,121,122,177,183,275,276,360,361]) or  (int(year[0][:4]) ==2021 and int(date[0]) in [1,44,46,92,93,95,96,121,122,139,274,287,359,361])
            # if int(year[0][:4]) == 2020 and int(date[0]) < 275:
            #     continue
            if day in [0,1,2,3,4,5]:
                if int(dow[0]) == day and not holiday:
                    day_list[day].append(i)
            else:
                if int(dow[0]) == 6 or holiday:
                    day_list[day].append(i)
    return day_list

def skip_outliner(input,output):
    select_list = []
    target_list = []
    for i, (data,target) in enumerate(zip(input,output)):
        if int(data[0][0][:4]) == 2020 and int(data[0][10]) in [129,140,142,144,145,147,150,151,152,153,216,217]:
            continue
        if int(data[0][0][:4]) == 2021 and int(data[0][10]) in [8,11]:
            continue
        select_list.append(i)
        target_list.append(i)
    return input[select_list],output[select_list]
    # np.save('input_no_outliner.npy',input[select_list])
    # np.save('output_no_outliner.npy',output[select_list])

def reassign_mean(input,output):
    # draw_data(input,output,2021,43)
    day_list = get_day_index(input[:,:,0],input[:,:,9],input[:,:,10])
    prototype=[]
    for i in range(7):
        mean_flow = output[day_list[i]].mean(0)
        # print(mean_flow)
        mean_flow_4 = mean_flow.repeat(4)
        # for data in mean_flow:
        #     mean_flow_4 += [float(data),float(data),float(data),float(data)] 
        input[day_list[i],:,8] = mean_flow_4
        prototype.append(mean_flow)
    # draw_data(input,output,2021,43)
    
    return input,output,prototype
    np.save('input_no_outliner_fix_allmean.npy',input)



def train_input_output():
    input_array1 = read_train_csv()
    everyday_a = []
    total_a = []
    before = datetime.strptime(input_array1[1][3], '%Y/%m/%d %H:%M').strftime('%Y-%m-%d')
    number = 0
    for item in input_array1[1:len(input_array1)]:
        now = datetime.strptime(item[3], '%Y/%m/%d %H:%M').strftime('%Y-%m-%d')
        # print(item)
        if now == before:
            flag = 1
            for obj in item:
                if obj == '':
                    flag = 0
                    break
            if flag == 1:
                everyday_a.append(item)
                number = number + 1
        else:
            if number < 96:
                everyday_a = []
            else:
                total_a.append(everyday_a)
                everyday_a = []
            before = now
            flag = 1
            for obj in item:
                if obj == '':
                    flag = 0
                    break
            if flag == 1:
                everyday_a.append(item)
                number = 1
            else:
                number = 0

    total_a.append(everyday_a)
    total_a = np.array(total_a)

    input_save = total_a[:, :, 0:11]
    #input_save = insert_prototype(input_save)

    # np.save("input_18months_imputed.npy", input_save)
    # loaddata = np.load("input_18months_imputed.npy")
    # print(loaddata.shape)

    output_array = total_a[:, :, 13]
    output_save = []
    item_save = []
    for item in output_array:
        new = float(item[0])
        i = 0
        for j in range(1, len(item)):
            if i < 3:
                new = new + float(item[j])
                i = i + 1
            else:
                new = new / 4
                item_save.append(new)
                i = 0
                new = float(item[j])
        item_save.append(new / 4)
        output_save.append(item_save)
        item_save = []

    output_save = np.array(output_save)
    # np.save("output_18months_imputed.npy", output_save)
    #
    # loaddata_2 = np.load("output_18months_imputed.npy")
    # print(loaddata_2.shape)

    input_save,output_save=skip_outliner(input_save,output_save)
    input_save,output_save,prototype=reassign_mean(input_save,output_save)




    return input_save, output_save,prototype



if __name__=='__main__':
    train_input, train_output, prototype = train_input_output()
    print(train_input.shape, train_output.shape)
    input=np.load('data/input_no_outliner_fix_allmean.npy',allow_pickle=True)
    print(train_input==input)
    test_data = test_data_preprocessing(prototype)
    print(test_data.shape)
    


