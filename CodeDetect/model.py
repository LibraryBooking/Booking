
from selenium.webdriver.common.by import By
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

import io
import time

import warnings
from selenium.webdriver.common.action_chains import ActionChains

import torch
import torch.nn as nn

from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image

import torch
import torch.nn as nn
import torch
import torchvision
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import PIL.ImageOps
print(torch.__version__)  #1.1.0
print(torchvision.__version__)  #0.3.0

import cv2 as cv
warnings.filterwarnings("ignore")

class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 16, kernel_size=3),
            nn.MaxPool2d((3, 3), stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),

            nn.ReflectionPad2d(1),
            nn.Conv2d(16, 8, kernel_size=5),
            nn.MaxPool2d((3, 3), stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 4, kernel_size=5),

            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(4),
        )

        self.Linear = nn.Sequential(
            nn.Linear(6400, 600),
            nn.LeakyReLU(inplace=True),

            nn.Linear(600, 200),
            nn.LeakyReLU(inplace=True),

            nn.Linear(200, 100),
            nn.ReLU(inplace=True),

            nn.Linear(100, 1)
        )

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)

        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        #   print(output1.size())

        #   print(outputR0.size())

        #     print(out.size())
        outputR = self.Linear(output1 - output2)
        return outputR

transform = transforms.Compose([transforms.Resize((48, 48)),  # 有坑，传入int和tuple有区别
                                transforms.ToTensor()])
class TestSet(Dataset):

    def __init__(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        index2 = index
        img_qian = Image.fromarray(cv.cvtColor(self.img_q,cv.COLOR_BGR2RGB))
        img_1 = Image.fromarray(cv.cvtColor(self.img1,cv.COLOR_BGR2RGB))
        img_2 = Image.fromarray(cv.cvtColor(self.img2,cv.COLOR_BGR2RGB))
        img_3 = Image.fromarray(cv.cvtColor(self.img3,cv.COLOR_BGR2RGB))


        img_qian = self.transform(img_qian)
        img_1 = self.transform(img_1)

        img_2 = self.transform(img_2)
        img_3 = self.transform(img_3)


        return img_qian,img_1,img_2,img_3

    def __len__(self):
        return 1
    def getImageFromNumpy(self,imgA,imgB,imgC,imgD):
        self.img_q=imgA
        self.img1=imgB
        self.img2=imgC
        self.img3=imgD
        pass

class getData():

    def __init__(self):
        self.iframeName = 'layui-layer-iframe1'
        self.seat_url='https://seat.lib.whu.edu.cn/login?targetUri=%2F'
        self.driver_path=r'D:\Microsoft Edge\chromedriver_win32\chromedriver.exe'#浏览器地址
        self.option=self.driver_option()
        self.driver= webdriver.Chrome(options=self.option, executable_path=self.driver_path, )
        self.SwitchToVerificatio()#转到验证码界面.需要判断是否完成加载
        time.sleep(1.5)
        pass

    def getArrPic(self):
        verfiBig = '/html/body/div[2]/div/div[1]/img'
        verfiSma = '/html/body/div[2]/div/div[2]/span/img'
        #     verfiRefresh='/html/body/div[2]/div/div[2]/div'

        self.BigEle = self.driver.find_element(By.XPATH, verfiBig)
        self.SmallEle = self.driver.find_element(By.XPATH, verfiSma)

        self.BigArr = self.getImageAsArray(self.BigEle.screenshot_as_png)[:, :, (2, 1, 0)]
        self.SmaArr = self.getImageAsArray(self.SmallEle.screenshot_as_png)[:, :, (2, 1, 0)]

        return self.BigArr


    def driver_option(self):
        #chrome_options = Options()
      #  chrome_options.add_argument('--headless')
       # chrome_options.add_argument('--disable-gpu')


        # 上面使用r是为了让字符串保持原有的意思 不转义字符

        options = webdriver.ChromeOptions()  # 此步骤很重要，设置为开发者模式，防止被各大网站识别出来使用了Selenium
        options.add_experimental_option('excludeSwitches', ['enable-automation'])  # 进行自动化伪装成 开发者模式
       # options.add_argument("--headless")  # 为Chrome配置无头模式
        options.add_argument('--disable-javascript')  # 禁用javascript
        options.add_experimental_option('prefs', {
            # "download.default_directory": 'D:\pdf文件22-1-6',                          # Change default directory for downloads
            "download.prompt_for_download": False,  # To auto download the file
            "download.directory_upgrade": True,
            "plugins.always_open_pdf_externally": True,  # It will not show PDF directly in chrome 修改谷歌配置 访问pdf文档时总是下载
           # "profile.managed_default_content_settings.images": 2,  # 禁止图片加载
            'useAutomationExtension': False
        })
        # 更换头部
        user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.82 Safari/537.36"
        )
        options.add_argument('user-agent=%s' % user_agent)
        return options

    def getImageAsArray(self,Bytes):
        bytePic=io.BytesIO(Bytes)
        PngImageFilePic=Image.open(bytePic,)

        PngAsArray=np.asarray(PngImageFilePic)#Generally RGBA format data
        return PngAsArray

    def SwitchToVerificatio(self):
        self.driver.get(self.seat_url)
        botton_click=self.driver.find_element(By.XPATH,'//*[@id="login"]/dd[3]/input')
        botton_click.click()

        self.driver.switch_to.frame(self.iframeName)
        pass




    def ImageProcessSma(self):
        self.SmaArr=cv.cvtColor(self.SmaArr,cv.COLOR_BGR2GRAY)


        qian=self.SmaArr[6:25,15:35]
        hou=self.SmaArr[6:25,37:55]#还需要一次灰度图二值化0 255
        mean=qian.mean()
        qian[qian>mean]=255
        qian[qian<255]=0

        mean = hou.mean()
        hou[hou > mean] = 255
        hou[hou < 255] = 0
        #二值化图像

        return qian,hou



    def Randomselect2(self,center):

        self.firstP=center[0]
        self.secondP=center[1]#中心点的格式

        pass
    def OrderClick2(self,center):
        self.Randomselect2(center)
        self.driver.save_screenshot('dianqian.png')
        ActionChains(self.driver).move_to_element_with_offset(self.BigEle,self.firstP[0],self.firstP[1]).click().move_to_element_with_offset(self.BigEle,self.secondP[0],self.secondP[1]).click().perform()
        #两次点击
        self.driver.save_screenshot('dianwan.png')

        pass

    def getNew(self):
        newelement=self.driver.find_element(By.XPATH,'/html/body/div[2]/div/div[2]/div')
        newelement.click()
        time.sleep(2)

    def Login(self,name=2020302021155,password=66666666):
        self.driver.switch_to.default_content()
        element_name=self.driver.find_element(By.XPATH,'//*[@id="login"]/dd[1]/input')
        element_password=self.driver.find_element(By.XPATH,'//*[@id="login"]/dd[2]/input')
        element_name.send_keys(name)
        element_password.send_keys(password)
        element_login=self.driver.find_element(By.XPATH,'//*[@id="login"]/dd[4]/input')
        element_login.click()

        time.sleep(10)
        pass

    def getMySeat(self):
        """
        找到我的位置,懒得写了
        """







def getPicCenter(BigArr,detected_position):
    '''
    返回(h,w,c) center
    '''
    tex=[0,1,2,3]
    hou=detected_position[:,4].astype('int')
    hou=hou.tolist()
    hou_index=hou.index(1)
    del tex[hou_index]
    center_hou=(int((detected_position[hou_index][0]+detected_position[hou_index][2])/2),int((detected_position[hou_index][1]+detected_position[hou_index][3])/2))

    Pic1=BigArr[detected_position[tex[0]][1]:detected_position[tex[0]][3],detected_position[tex[0]][0]:detected_position[tex[0]][2],:]
    center1=(int((detected_position[tex[0]][0]+detected_position[tex[0]][2])/2),int((detected_position[tex[0]][1]+detected_position[tex[0]][3])/2))

    Pic2=BigArr[detected_position[tex[1]][1]:detected_position[tex[1]][3],detected_position[tex[1]][0]:detected_position[tex[1]][2],:]
    center2=(int((detected_position[tex[1]][0]+detected_position[tex[1]][2])/2),int((detected_position[tex[1]][1]+detected_position[tex[1]][3])/2))

    Pic3=BigArr[detected_position[tex[2]][1]:detected_position[tex[2]][3],detected_position[tex[2]][0]:detected_position[tex[2]][2],:]
    center3=(int((detected_position[tex[2]][0]+detected_position[tex[2]][2])/2),int((detected_position[tex[2]][1]+detected_position[tex[2]][3])/2))


    return Pic1,Pic2,Pic3,(center1,center2,center3,center_hou)