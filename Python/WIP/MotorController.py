import RPi.GPIO as PIN
from time import sleep
PIN.setmode(PIN.BCM)
PIN.setwarning(False)

Ena=2
IN1=3
IN2=4

PIN.setup(Ena,PIN.OUT)
PIN.setup(IN1,PIN.OUT)
PIN.setup(IN2,PIN.OUT)
pwmA=PIN.PWN(Ena,100);
pwmA.start(0);


pwmA.ChargeDutyCycle(60);
PIN.output(IN1,PIN.LOW)
PIN.output(IN2,PIN.HIGH)
sleep(2)
pwmA.ChargeDutyCycle(0 )
