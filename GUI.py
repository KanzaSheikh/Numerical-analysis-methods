from tkinter import *
import math
from sympy import *
import numpy as np
import numpy
import fpdf
from datetime import datetime

# specifications of the GUI

expr = ""
y = ""
yprime = ""
data=[]
filename = datetime.now().strftime("%Y%m%d-%H%M%S")

#------------------------------------------------------------------------------------------------------------------------/

def WritePdf(data,names):  
    pdf = fpdf.FPDF(format='letter')
    pdf.add_page()
    pdf.set_font("Arial", size=12)   
    for i in data:
        pdf.write(5,str(i))
    name=str(names)+".pdf"
    pdf.output(name)
    
    print("PDF FILE Writing complete") 
    
#------------------------------------------------------------------------------------------------------------------------/

def decide_method(number): # lambda:decide_method(1)
    global method_number
    method_number = number
    print(method_number)
    
#------------------------------------------------------------------------------------------------------------------------/
    
def derivative(a):
    x = Symbol('x')     #specifying the variable used in the equation
    yprime = a.diff(x)
    return yprime

#------------------------------------------------------------------------------------------------------------------------/
    
def integral(a):
    x = Symbol('x')     #specifying the variable used in the equation
    yprime = a. integrate(x)
    return yprime

#------------------------------------------------------------------------------------------------------------------------/
def direct_method():
    if(method_number==1):
        NewtonRaphson()
    elif(method_number==2):
        FixedPointIteration()
    elif(method_number==3):
        secant()
    elif(method_number==4):
        bisection()
    elif(method_number==5):
        RegulaFalsi()
    elif(method_number==6):
        NewtonDifferenceInt2D()
    elif(method_number==7):
        CompositeTrapezoidal()
    elif(method_number==8):
        CompositeSimpsons()
    elif(method_number==9):
        CompositeSimpsons3_8()
 
#------------------------------------------------------------------------------------------------------------------------/
        
def NewtonRaphson():
    global expr
    global y
    global enter_equation_text
    expr = enter_equation_text.get()
    x = Symbol('x')      #specifying the variable used in the equation
    y = eval(expr)
    yprime=derivative(y)
    x0= init_aprrox_text.get()
    xn = float(x0)		
    TOL = float(tolerance1_text.get())
    func = lambdify(x, y, "math")		#converting the sympy function to numeric function
    dfunc = lambdify(x, yprime, "math")	#converting the sympy function to numeric function
    
    for i in range(0,100):    
        x = float(xn)
        xn = float(x - func(x)/dfunc(x)) 	#perform the required calculaton for the method
        err = abs(xn-x)
        data.append("after ")
        data.append(str(i+1))
        data.append(" iterations, the approximate root = ")
        data.append(str(xn))
        data.append("\n")
        data.append(" f(x) = ")
        data.append(str(func(x)))
        data.append( " Error = ")
        data.append(str(err))
        data.append("\n")
        print("After " + str(i+1) + " iterations, the approximate root = " + str(xn))
        print( " f(x) = " + str(func(x)) + " Error = " + str(err))

        if(err<TOL): 	#condition to break loop if error is less then error tolerance level
            break
    if(err<TOL):
        print("The desired accuracy achieved; Solution converged.")
        con = "The desired accuracy achieved; Solution converged."
    else:
        print("The maximum number of iterations exceeded the limit.")
        con = "The maximum number of iterations exceeded the limit."
    data.append("\n")
    data.append(con)
    WritePdf(data,filename)
        
#------------------------------------------------------------------------------------------------------------------------/
        
def FixedPointIteration():
    global expr
    global y
    global enter_equation_text
    expr = enter_equation_text.get()
    x = Symbol('x')      #specifying the variable used in the equation
    y = eval(expr)
#    find Integral
    yprime=integral(y)
    x0= init_aprrox_text.get()
    xn = float(x0)		
    TOL = float(tolerance1_text.get())	
    func = lambdify(x, y, "math")		#converting the sympy function to numeric function
    dfunc = lambdify(x, yprime, "math")	#converting the sympy function to numeric function
    
    for i in range(0,100):    
        x = float(xn)
        xn = float(x - func(x)/dfunc(x)) 	#perform the required calculaton for the method
        err = abs(xn-x)
        data.append("after ")
        data.append(str(i+1))
        data.append(" iterations, the approximate root = ")
        data.append(str(xn))
        data.append("\n")
        data.append(" f(x) = ")
        data.append(str(func(x)))
        data.append( " Error = ")
        data.append(str(err))
        data.append("\n")
        print("After " + str(i+1) + " iterations, the approximate root = " + str(xn))
        print( " f(x) = " + str(func(x)) + " Error = " + str(err))
        if(err<TOL): 	#condition to break loop if error is less then error tolerance level
            break
    if(err<TOL):
        print("The desired accuracy achieved; Solution converged.")
        con = "The desired accuracy achieved; Solution converged."
    else:
        print("The maximum number of iterations exceeded the limit.")
        con = "The maximum number of iterations exceeded the limit."
    data.append("\n")
    data.append(con)
    WritePdf(data,filename)

#------------------------------------------------------------------------------------------------------------------------/
        
def secant():  
    global expr
    global y
    global enter_equation_text
    expr = enter_equation_text.get()
    x = Symbol('x')      #specifying the variable used in the equation
    y = eval(expr)
    print("Approximate solution of f(x)=0 on interval [a,b] by the secant method. \n Parameters \n x0,x1 : numbers \n The interval in which to search for a solution. The function returns \n None if f(a)*f(b) >= 0 since a solution is not guaranteed. \n n : (positive) integer \n The number of iterations to implement.")
    data.append("Approximate solution of f(x)=0 on interval [a,b] by the secant method. \n Parameters \n x0,x1 : numbers \n The interval in which to search for a solution. The function returns \n None if f(a)*f(b) >= 0 since a solution is not guaranteed. \n n : (positive) integer \n The number of iterations to implement.")
    data.append("\n")
    a = int(starting_interval_text.get())	
    b = int(ending_interval_text.get())	
    n = int(number_of_iterations_text.get())
    TOL = float(tolerance2_text.get()) 	#input for the error tolerance
    yd = derivative(y)	#calculating the derivative of tx+2*xhe the equation

    dfunc = lambdify(x, yd, "math")
    fx0=dfunc(a)
    fx1=dfunc(b)
    print('Derivative of  equation at ',a,' is ',fx0) # Evaluating f(x) at x0
    data.append('Derivative of  equation at ')
    data.append(str(a))
    data.append(" is ")
    data.append(str(fx0))
    data.append("\n")
    print('Derivative of  equation at ',b,' is ',fx1) # Evaluating f(x) at x1
    data.append('Derivative of  equation at ')
    data.append(str(b))
    data.append(' is ')
    data.append(str(fx1))
    data.append("\n")
    xn=b
    x0=a
    x1=b							
    for k in range(2,n+1):
        xp=xn
        xn = x1 - (fx1*(x1-x0))/(fx1-fx0)
       # xn=x1-(fx1*(x1–x0))/(fx1–fx0)
        
        fxn=dfunc(xn)
        err=abs(xn-xp)/abs(xn)
        #err=abs(xn–xp)/abs(xn)
        data.append("after ")
        data.append(str(k-1))
        data.append(" iterations, the approximate root = ")
        data.append(str(xn))
        data.append("\n")
        data.append(" f(x) = ")
        data.append(str(fxn))
        data.append( " Error = ")
        data.append(str(err))
        data.append("\n")
        print("After " + str(k-1) + " iterations, the approximate root = " + str(xn))
        print( " f(x) = " + str(fxn) + " Error = " + str(err))
        if(err<TOL): 	#condition to break loop if error is less then error tolerance level
            break
        else:
            x0 = x1
            fx0 = fx1 																																													
            x1 = xn 																							
            fx1 = fxn	        
    if(err<TOL):
        print("The desired accuracy achieved; Solution converged.")
        con = "The desired accuracy achieved; Solution converged."
    else:
        print("The maximum number of iterations exceeded the limit.")
        con = "The maximum number of iterations exceeded the limit."
    data.append("\n")
    data.append(con)
    WritePdf(data,filename)

#------------------------------------------------------------------------------------------------------------------------/
        
def bisection():  
    global expr
    global y
    global enter_equation_text
    expr = enter_equation_text.get()
    x = Symbol('x')      #specifying the variable used in the equation
    y = eval(expr)
    print("Approximate solution of f(x)=0 on interval [a,b] by the bisection Method . \n Parameters \n x0,x1 : numbers \n The interval in which to search for a solution. The function returns \n None if f(a)*f(b) >= 0 since a solution is not guaranteed. \n n : (positive) integer \n The number of iterations to implement.")
    data.append("Approximate solution of f(x)=0 on interval [a,b] by the secant method. \n Parameters \n x0,x1 : numbers \n The interval in which to search for a solution. The function returns \n None if f(a)*f(b) >= 0 since a solution is not guaranteed. \n n : (positive) integer \n The number of iterations to implement.")
    data.append("\n")
#   '''
#    Approximate solution of f(x)=0 on interval [a,b] by the bisection Method .
#    Parameters
#    f : function
#        The function for which we are trying to approximate a solution f(x)=0.
#    a,b : numbers
#        The interval in which to search for a solution. The function returns
#        None if f(a)*f(b) >= 0 since a solution is not guaranteed.
#    n : (positive) integer
#        The number of iterations to implement.
#    '''
    a = int(starting_interval_text.get())	
    b = int(ending_interval_text.get())	
    n = int(number_of_iterations_text.get())
    TOL = float(tolerance2_text.get()) 	#input for the error tolerance
    yd = derivative(y)	#calculating the derivative of tx+2*xhe the equation
    print('Derivative of  equation',yd)
    dfunc = lambdify(x, yd, "math")
    xn = b							
    x0 = a								
    x1 = b								
    fx0=dfunc(a)
    fx1=dfunc(b)
    print('Derivative of  equation at ',a,' is ',fx0) # Evaluating f(x) at x0
    data.append('Derivative of  equation at ')
    data.append(str(a))
    data.append(" is ")
    data.append(str(fx0))
    data.append("\n")
    print('Derivative of  equation at ',b,' is ',fx1) # Evaluating f(x) at x	
    data.append('Derivative of  equation at ')
    data.append(str(b))
    data.append(' is ')
    data.append(str(fx1))
    data.append("\n")																															
    for k in range(2,n+1):																													
        xp = xn
        xn = x0 + (x1 - x0 ) /2		
        fxn=dfunc(xn)
        err=abs(xn-xp)/abs(xn)
        data.append("after ")
        data.append(str(k-1))
        data.append(" iterations, the approximate root = ")
        data.append(str(xn))
        data.append("\n")
        data.append(" f(x) = ")
        data.append(str(fxn))
        data.append( " Error = ")
        data.append(str(err))
        data.append("\n")
        print("After " + str(k-1) + " iterations, the approximate root = " + str(xn))
        print( " f(x) = " + str(fxn) + " Error = " + str(err))
        
																																
        if ( fxn < TOL ):     
            break																	
        elif( err < TOL ):
            break																	
        elif ( fx0*fxn < 0 ):
            x1=xn
            fx1 = fxn
        else:																												
            x0 = xn
            fx0 = fxn        
    if(err<TOL):
        print("The desired accuracy achieved; Solution converged.")
        con = "The desired accuracy achieved; Solution converged."
    else:
        print("The maximum number of iterations exceeded the limit.")
        con = "The maximum number of iterations exceeded the limit."
    data.append("\n")
    data.append(con)
    WritePdf(data,filename)
        
#------------------------------------------------------------------------------------------------------------------------/
        
def RegulaFalsi():
    global expr
    global y
    global enter_equation_text
    expr = enter_equation_text.get()
    x = Symbol('x')      #specifying the variable used in the equation
    y = eval(expr)
    print("Approximate solution of f(x)=0 on interval [x0,x1] by the RegulaFalsi Method . \n Parameters \n x0,x1 : numbers \n The interval in which to search for a solution. The function returns \n None if f(a)*f(b) >= 0 since a solution is not guaranteed. \n n : (positive) integer \n The number of iterations to implement.")
    data.append("Approximate solution of f(x)=0 on interval [a,b] by the secant method. \n Parameters \n x0,x1 : numbers \n The interval in which to search for a solution. The function returns \n None if f(a)*f(b) >= 0 since a solution is not guaranteed. \n n : (positive) integer \n The number of iterations to implement.")
    data.append("\n")
    
    
#   '''
#    Approximate solution of f(x)=0 on interval [a,b] by the RegulaFalsi Method .
#    Parameters
#    f : function
#        The function for which we are trying to approximate a solution f(x)=0.
#    a,b : numbers
#        The interval in which to search for a solution. The function returns
#        None if f(a)*f(b) >= 0 since a solution is not guaranteed.
#    n : (positive) integer
#        The number of iterations to implement.
#    '''
    a = int(starting_interval_text.get())	
    b = int(ending_interval_text.get())	
    n = int(number_of_iterations_text.get())
    TOL = float(tolerance2_text.get()) 	#input for the error tolerance
    yd = derivative(y)	#calculating the derivative of tx+2*xhe the equation
    print('Derivative of  equation',yd)
    dfunc = lambdify(x, yd, "math")
    xn = b							
    x0 = a								
    x1 = b								
    fx0=dfunc(a)
    fx1=dfunc(b)
    print('Derivative of  equation at ',a,' is ',fx0) # Evaluating f(x) at x0
    data.append('Derivative of  equation at ')
    data.append(str(a))
    data.append(" is ")
    data.append(str(fx0))
    data.append("\n")
    print('Derivative of  equation at ',b,' is ',fx1) # Evaluating f(x) at x	
    data.append('Derivative of  equation at ')
    data.append(str(b))
    data.append(' is ')
    data.append(str(fx1))
    data.append("\n")																																
    for k in range(2,n+1):																													
        xp = xn
        xn = x1 - (fx1 * (x1 - x0)) / (fx1 - fx0)
        fxn=dfunc(xn)							
        err=abs(xn-xp)/abs(xn)
        data.append("after ")
        data.append(str(k-1))
        data.append(" iterations, the approximate root = ")
        data.append(str(xn))
        data.append("\n")
        data.append(" f(x) = ")
        data.append(str(fxn))
        data.append( " Error = ")
        data.append(str(err))
        data.append("\n")
        print("After " + str(k-1) + " iterations, the approximate root = " + str(xn))
        print( " f(x) = " + str(fxn) + " Error = " + str(err))       																																
        if ( fxn < TOL ):     
            break																	
        elif( err < TOL ):
            break																	
        elif ( fx0*fxn < 0 ):
            x1=xn
            fx1 = fxn
        else:																												
            x0 = xn
            fx0 = fxn        
    if(err<TOL):
        print("The desired accuracy achieved; Solution converged.")
        con = "The desired accuracy achieved; Solution converged."
    else:
        print("The maximum number of iterations exceeded the limit.")
        con = "The maximum number of iterations exceeded the limit."
    data.append("\n")
    data.append(con)
    WritePdf(data,filename)	

#------------------------------------------------------------------------------------------------------------------------/

def NewtonDifferenceInt2D():
    n=2
    ddf=numpy.zeros((n+1,n+1))
    f=numpy.array([5, 10, 12])
    x=numpy.array([0, 1, 3])
    
    xp = int(real_value_text.get())	
    						
    for i in range(0,n):
        ddf[i][0] = f[i]																						
    													
    for j in range(1,n):																						
    	for i in range(0,n-j):
    		ddf[i][j] = ( ddf[i+1][j-1] - ddf[i][j-1] ) / ( x[i+j] - x[i] )							
    						
    pro = 1
    fxp = ddf[0][0] ;																			
    for k in range(1,n):   
    	pro = pro * ( xp - x[k-1] )																				
    	fxp = fxp + pro * ddf[0][k]																		
    data.append("The interpolate or extrapolate value of function at x = ")	
    data.append(xp)
    data.append(" is ")
    data.append(fxp)																				
    print("The interpolate or extrapolate value of function at x = ",xp," is ",fxp)
    WritePdf(data,filename)	

#------------------------------------------------------------------------------------------------------------------------/
    
def CompositeTrapezoidal():
    global y
    global enter_equation_text
    x = Symbol('x')      #specifying the variable used in the equation
    y = eval(enter_equation_text.get())
    
    x0 = float(composite_lower_limit_text.get())
    xn = float(composite_upper_limit_text.get())
    n = int(composite_subintervals_text.get())
	
    func = lambdify(x, y, "math")		#converting the sympy function to numeric function
    

    h = (xn - x0)/n 
    I = func(x0) + func(xn)
    xc = x0 
    sum = 0.0 

    for j in range(1,n):
        xc = xc + h
        sum = sum + func(xc) 

    I = (h / 2.0)  *  (I + 2.0 * sum) 

    data.append("The approximate integral = ")
    data.append(I)
    print("The approximate integral = ", I)
    WritePdf(data,filename)	
    
#------------------------------------------------------------------------------------------------------------------------/
    
def CompositeSimpsons():
    global y
    global enter_equation_text
    x = Symbol('x')      #specifying the variable used in the equation
    y = eval(enter_equation_text.get())
    
    x0 = float(composite_lower_limit_text.get())
    xn = float(composite_upper_limit_text.get())
    n = int(composite_subintervals_text.get())
	
    func = lambdify(x, y, "math")		#converting the sympy function to numeric function
    

    h = (xn - x0)/n 
    I = func(x0) + func(xn)
    xc = x0 
    sum1 = 0.0
    sum2 = 0.0

    for j in range(1,n):
        xc = xc + h
        if ( j%2 != 0):																									
            sum1 = sum1 + func(xc) 																					
        else:
            sum2 = sum2 + func(xc)

    I = (h / 3.0) * (I + 4 * sum1 + 2 * sum2)

    data.append("The approximate integral = ")
    data.append(I)
    print("The approximate integral = ", I)
    WritePdf(data,filename)

#------------------------------------------------------------------------------------------------------------------------/

def CompositeSimpsons3_8():
    global y
    global enter_equation_text
    x = Symbol('x')      #specifying the variable used in the equation
    y = eval(enter_equation_text.get())
    
    x0 = float(composite_lower_limit_text.get())
    xn = float(composite_upper_limit_text.get())
    n = int(composite_subintervals_text.get())
	
    func = lambdify(x, y, "math")		#converting the sympy function to numeric function
    

    h = (xn - x0)/n 
    I = func(x0) + func(xn)
    xc = x0 
    sum1 = 0.0
    sum2 = 0.0

    for j in range(1,n):
        xc = xc + h
        if ( j%3 == 0):																									
            sum2 = sum2 + func(xc) 																					
        else:
            sum1 = sum1 + func(xc)

    I = (3.0 * h / 8.0) * (I + 2 * sum2 + 3 * sum1)

    data.append("The approximate integral = ")
    data.append(I)
    print("The approximate integral = ", I)
    WritePdf(data,filename)
    
#------------------------------------------------------------------------------------------------------------------------/
    
def next_output():
    global method_number
    window = Tk()
    window.geometry("354x460")
    window.title("Input Parameters")
    
    if(method_number==1 or method_number==2):
        global init_aprrox_text
        global tolerance1_text
        
        if(method_number==1):
              windowlabel1 = Label(window, text="Welcome to \n Newton Rapson Method",fg='black', font=("Helvetica", 20))
              windowlabel1.pack(side=TOP)
              
        elif(method_number==2):
            windowlabel1 = Label(window, text="Welcome to \nFixed Point Iteration Method",fg='black', font=("Helvetica", 20))
            windowlabel1.pack(side=TOP)
            
        init_aprrox_label = Label(window, text="Enter the initial approximation",fg='black', font=("Helvetica", 15))
        init_aprrox_label.pack(side=TOP)
        textin = StringVar()
        init_aprrox_text = Entry(window, font=("Courier New", 12, 'bold'), textvar=textin, width=25, bd=5)
        init_aprrox_text.pack()
        tolerance1_label = Label(window, text="Enter the error tolerance",fg='black', font=("Helvetica", 15))
        tolerance1_label.pack(side=TOP)
        textin = StringVar()
        tolerance1_text = Entry(window, font=("Courier New", 12, 'bold'), textvar=textin, width=25, bd=5)
        tolerance1_text.pack()
        go_but = Button(window, padx=14, pady=12, bd=4, bg='white',  text="Go", font=("Helvetica", 16, 'bold'), command=direct_method)
        go_but.place(x=250, y=380)
        
            
    elif(method_number==3 or method_number==4 or method_number==5):
        global starting_interval_text
        global ending_interval_text
        global number_of_iterations_text
        global tolerance2_text
        
        if(method_number==3):
            windowlabel1 = Label(window, text="Welcome to \nSecant Method",fg='black', font=("Helvetica", 20))
            windowlabel1.pack(side=TOP)
        elif(method_number==4):
            windowlabel1 = Label(window, text="Welcome to \nBisection Method",fg='black', font=("Helvetica", 20))
            windowlabel1.pack(side=TOP)
        elif(method_number==5):
            windowlabel1 = Label(window, text="Welcome to \nRegulaFalsi Method",fg='black', font=("Helvetica", 20))
            windowlabel1.pack(side=TOP)
        starting_interval_label = Label(window, text="Enter The starting interval in which to search for a solution x0: ",fg='black', font=("Helvetica", 10))
        starting_interval_label.pack(side=TOP)
        textin = StringVar()
        starting_interval_text = Entry(window, font=("Courier New", 12, 'bold'), textvar=textin, width=25, bd=5)
        starting_interval_text.pack()
        ending_interval_label = Label(window, text="Enter The ending interval in which to search for a solution x1: ",fg='black', font=("Helvetica", 10))
        ending_interval_label.pack(side=TOP)
        textin = StringVar()
        ending_interval_text = Entry(window, font=("Courier New", 12, 'bold'), textvar=textin, width=25, bd=5)
        ending_interval_text.pack()
        number_of_iterations_label = Label(window, text="Enter (positive) integer The number of iterations to implement N: ",fg='black', font=("Helvetica", 10))
        number_of_iterations_label.pack(side=TOP)
        textin = StringVar()
        number_of_iterations_text = Entry(window, font=("Courier New", 12, 'bold'), textvar=textin, width=25, bd=5)
        number_of_iterations_text.pack()
        tolerance2_label = Label(window, text="Enter the error tolerance",fg='black', font=("Helvetica", 10))
        tolerance2_label.pack(side=TOP)
        textin = StringVar()
        tolerance2_text = Entry(window, font=("Courier New", 12, 'bold'), textvar=textin, width=25, bd=5)
        tolerance2_text.pack()
        go_but = Button(window, padx=14, pady=12, bd=4, bg='white',  text="Go", font=("Helvetica", 16, 'bold'), command=direct_method)
        go_but.place(x=250, y=380)
    
           
    elif(method_number==6):
        global real_value_text
        windowlabel1 = Label(window, text="Welcome to \nNewtonDifferenceInt 2nd Degree Method",fg='black', font=("Helvetica", 20))
        windowlabel1.pack(side=TOP)
        real_value_label = Label(window, text="Enter a real value it which the interpolation is to be obtained:",fg='black', font=("Helvetica", 10))
        real_value_label.pack(side=TOP)
        textin = StringVar()
        real_value_text = Entry(window, font=("Courier New", 12, 'bold'), textvar=textin, width=25, bd=5)
        real_value_text.pack()
        go_but = Button(window, padx=14, pady=12, bd=4, bg='white',  text="Go", font=("Helvetica", 16, 'bold'), command=NewtonDifferenceInt2D)
        go_but.place(x=250, y=380)
        
    elif(method_number==7 or method_number==8 or method_number==9):
        global composite_lower_limit_text
        global composite_upper_limit_text
        global composite_subintervals_text
        
        if(method_number==7):
            windowlabel1 = Label(window, text="Welcome to \nComposite Trapezoidal Method",fg='black', font=("Helvetica", 20))
            windowlabel1.pack(side=TOP)
        elif(method_number==8):
            windowlabel1 = Label(window, text="Welcome to \nComposite Simpson's Method",fg='black', font=("Helvetica", 20))
            windowlabel1.pack(side=TOP)
        elif(method_number==9):
            windowlabel1 = Label(window, text="Welcome to \nComposite Simpson's 3/8 Method",fg='black', font=("Helvetica", 20))
            windowlabel1.pack(side=TOP)
            
        windowlabel1 = Label(window, text="Welcome to \nComposite Trapezoidal Method",fg='black', font=("Helvetica", 20))
        windowlabel1.pack(side=TOP)
        composite_lower_limit_label = Label(window, text="Enter the lower limit of the integral: ",fg='black', font=("Helvetica", 10))
        composite_lower_limit_label.pack(side=TOP)
        textin = StringVar()
        composite_lower_limit_text = Entry(window, font=("Courier New", 12, 'bold'), textvar=textin, width=25, bd=5)
        composite_lower_limit_text.pack()
        composite_upper_limit_label = Label(window, text="Enter the upper limit of the integral: ",fg='black', font=("Helvetica", 10))
        composite_upper_limit_label.pack(side=TOP)
        textin = StringVar()
        composite_upper_limit_text = Entry(window, font=("Courier New", 12, 'bold'), textvar=textin, width=25, bd=5)
        composite_upper_limit_text.pack()
        composite_subintervals_label = Label(window, text="Enter the number of subintervals n: ",fg='black', font=("Helvetica", 10))
        composite_subintervals_label.pack(side=TOP)
        textin = StringVar()
        composite_subintervals_text = Entry(window, font=("Courier New", 12, 'bold'), textvar=textin, width=25, bd=5)
        composite_subintervals_text.pack()
        go_but = Button(window, padx=14, pady=12, bd=4, bg='white',  text="Go", font=("Helvetica", 16, 'bold'), command=direct_method)
        go_but.place(x=250, y=380)
    
#------------------------------------------------------------------------------------------------------------------------/    
    
def init_interface():
    me = Tk()
    me.geometry("354x460")
    me.title("Methods")
    global enter_equation_text
    enter_equation_label = Label(me, text="Enter the equation",fg='black', font=("Helvetica", 20))
    enter_equation_label.pack(side=TOP)
    #me.config(background='black')
    textin = StringVar()
    enter_equation_text = Entry(me, font=("Courier New", 12, 'bold'), textvar=textin, width=25, bd=5, bg='white')
    enter_equation_text.pack()
    choose_method_label = Label(me, text="Choose the method",fg='black', font=("Helvetica", 15))
    choose_method_label.pack(side=TOP)
    choose_method_label.place(x=10,y=70)
    newtonraphson_check = Checkbutton(me, text="Newton-Raphson method.", command=lambda: decide_method(1))
    newtonraphson_check.place(x=10,y=100)
    fixedpoint_check=Checkbutton(me, text="Fixed-Point Iteration method", command=lambda: decide_method(2))
    fixedpoint_check.place(x= 10, y=130)
    secant_check=Checkbutton(me, text="Secant method", command=lambda: decide_method(3))
    secant_check.place(x= 10, y=160)
    bisection_check=Checkbutton(me, text="Bisection method", command=lambda: decide_method(4))
    bisection_check.place(x= 10, y=190)
    regulafalsi_check=Checkbutton(me, text="Regula-Falsi method", command=lambda: decide_method(5))
    regulafalsi_check.place(x= 10, y=220)
    newtondivideddifference_check=Checkbutton(me, text="Newton’s Divided Difference Interpolation", command=lambda: decide_method(6))
    newtondivideddifference_check.place(x= 10, y=250)
    compositetrapezoid_check=Checkbutton(me, text="Composite Trapezoidal rule", command=lambda: decide_method(7))
    compositetrapezoid_check.place(x= 10, y=280)
    compositesimpson_check=Checkbutton(me, text="Composite Simpson’s", command=lambda: decide_method(8))
    compositesimpson_check.place(x= 10, y=310)
    compositesimpson3_8_check=Checkbutton(me, text="Composite Simpson’s 3/8 rule",  command=lambda: decide_method(9))
    compositesimpson3_8_check.place(x= 10, y=340)
    next_but = Button(me, padx=14, pady=12, bd=4, bg='white',  text="Next", font=("Helvetica", 16, 'bold'), command=next_output)
    next_but.place(x=250, y=380)
    me.mainloop()

#------------------------------------------------------------------------------------------------------------------------/
    

init_interface()