# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 00:09:26 2020

@author: Kanza Sheikh
"""


import math
from sympy import *
import numpy as np
import numpy
import cmath

#input equation
def Equation_input():
    try:
        print("input equation as \n ' x+2*x ' \n ' x+3*x+x**2 ' \n  ' x+2*x+sin(0.2) ' \n ")
        print("\n *************************************************************************** \n")
        expr = input("Enter the Equation (in terms of x):")  #taking an equation as a string input
        x = Symbol('x')     #specifying the variable used in the equation
        y = eval(expr)      #evaluating the input string
        print('You Enteted',y)
        return y
    except ValueError:
        print('You entered an invalid Equation')
        
#calculating the derivative of the the equation
def derivative(a):
    x = Symbol('x')     #specifying the variable used in the equation
    yprime = a.diff(x)
    print('Derivative of  equation',yprime)
    return yprime
#calculating the integral of the the equation
def integral(a):
    x = Symbol('x')     #specifying the variable used in the equation
    yprime = a. integrate(x)
    print('integral of  equation',yprime)
    return yprime


def NewtonRapson():
    print("\n *************************************************************************** \n")
    print('Welcome to Newton Rapson Method')
    print("\n *************************************************************************** \n")
#    Input Equation
    y=Equation_input()
#    find derivative
    yprime=derivative(y)
    x = Symbol('x')     #specifying the variable used in the equation
    x0 = float(input("Enter the initial approximation x0 : "))	#input for the initial approximation	
    TOL = input("Enter the error tolerance : ")	#input for the error tolerance
    TOL = eval(TOL)
    print('You enter error tolerance',TOL)
    xn = float(x0)		
    func = lambdify(x, y, "math")		#converting the sympy function to numeric function
    dfunc = lambdify(x, yprime, "math")	#converting the sympy function to numeric function
    
    for i in range(0,100):    
        x = float(xn)
        xn = float(x - func(x)/dfunc(x)) 	#perform the required calculaton for the method
        err = abs(xn-x)
        print("After " + str(i+1) + " iterations, the approximate root = " + str(xn))
        print( " f(x) = " + str(func(x)) + " Error = " + str(err))
        if(err<TOL): 	#condition to break loop if error is less then error tolerance level
            break
    if(err<TOL):
        print("The desired accuracy achieved; Solution converged.")
    else:
        print("The maximum number of iterations exceeded the limit.")

def FixedPointIteration():
    print("\n *************************************************************************** \n")
    print('Welcome to fixed point iteration method')
    print("\n *************************************************************************** \n")
#    Input Equation
    y=Equation_input()
    x0 = float(input("Enter the initial approximation x0 : "))	#input for the initial approximation	
    TOL = input("Enter the error tolerance : ")	#input for the error tolerance
    TOL = eval(TOL)
    xn = float(x0)		
    
    for i in range(0,100):    
        x = float(xn)
        xn = float(0.25*(math.exp(x)-math.sin(x))) 	#perform the required calculaton for the method
        err = abs(xn-x)
        print("After " + str(i+1) + " iterations, the approximate root = " + str(xn))
        print( " exp(x) = " + str(math.exp(x)) + "sin(x) = " + str(math.sin(x)) + "Error = " + str(err))
        if(err<TOL): 	#condition to break loop if error is less then error tolerance level
            break
    if(err<TOL):
        print("The desired accuracy achieved; Solution converged.")
    else:
        print("The maximum number of iterations exceeded the limit.")

def secant():   
    print("\n *************************************************************************** \n")
    print('Welcome to Secant Method')
    print("\n *************************************************************************** \n")

#  Input Equation
    f=Equation_input()
    print("Approximate solution of f(x)=0 on interval [a,b] by the secant method. \n Parameters \n x0,x1 : numbers \n The interval in which to search for a solution. The function returns \n None if f(a)*f(b) >= 0 since a solution is not guaranteed. \n n : (positive) integer \n The number of iterations to implement.")

    a = float(input("Enter The starting interval in which to search for a solution x0: "))	
    b = float(input("Enter The ending interval in which to search for a solution x1: "))	
    n = int(input("Enter (positive) integer The number of iterations to implement N: "))
    TOL = input("Enter the error tolerance : ")	#input for the error tolerance
    TOL = eval(TOL)
    print('You enter error tolerance',TOL)
    x = Symbol('x') 
    func = lambdify(x, f, "math")

    xn=b
    x0=a
    x1=b							
    for k in range(2,n+1):
        xp=xn
        xn = x1 - (func(x1)*(x1-x0))/(func(x1)-func(x0))
        err=abs(xn-xp)/abs(xn)
        print("After " + str(k-1) + " iterations, the approximate root = " + str(xn))
        print( " f(x) = " + str(func(xn)) + " Error = " + str(err))
        if(err<TOL): 	#condition to break loop if error is less then error tolerance level
            break
        else:
            x0 = x1
            x1 = xn 																							
    if(err<TOL):
        print("The desired accuracy achieved; Solution converged.")
    else:
        print("The maximum number of iterations exceeded the limit.")

def bisection():  
    print("\n *************************************************************************** \n")
    print('Welcome to bisection Method')
    print("\n *************************************************************************** \n")  
    #  Input Equation
    f=Equation_input()
    print("Approximate solution of f(x)=0 on interval [a,b] by the bisection Method . \n Parameters \n x0,x1 : numbers \n The interval in which to search for a solution. The function returns \n None if f(a)*f(b) >= 0 since a solution is not guaranteed. \n n : (positive) integer \n The number of iterations to implement.")

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
    a = float(input("Enter The starting interval in which to search for a solution x0: "))	
    b = float(input("Enter The ending interval in which to search for a solution x1: "))	
    n = int(input("Enter (positive) integer The number of iterations to implement N: "))
    TOL = input("Enter the error tolerance : ")	#input for the error tolerance
    TOL = eval(TOL) 
    x = Symbol('x') 
    func = lambdify(x, f, "math")
    xn = b							
    x0 = a								
    x1 = b																																							
    for k in range(2,n+1):																													
        xp = xn
        xn = (x0 + x1) /2		
        #fxn=func(xn)
        err=abs(xn-xp)/abs(xn)
        print("After " + str(k-1) + " iterations, the approximate root = " + str(xn))
        print( " f(x) = " + str(func(xn)) + " Error = " + str(err))
        
																																
        if ( fxn < TOL ):     
            break																	
        elif( err < TOL ):
            break																	
        elif ( func(x0)*func(xn) < 0 ):
            x1=xn

        else:																												
            x0 = xn
    if(err<TOL):
        print("The desired accuracy achieved; Solution converged.")
    else:
        print("The maximum number of iterations exceeded the limit.")

def RegulaFalsi():
    print("\n *************************************************************************** \n")
    print('Welcome to RegulaFalsi Method')
    print("\n *************************************************************************** \n")

 #  Input Equation
    f=Equation_input()
    print("Approximate solution of f(x)=0 on interval [x0,x1] by the RegulaFalsi Method . \n Parameters \n x0,x1 : numbers \n The interval in which to search for a solution. The function returns \n None if f(a)*f(b) >= 0 since a solution is not guaranteed. \n n : (positive) integer \n The number of iterations to implement.")
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
    a = float(input("Enter The starting interval in which to search for a solution x0: "))	
    b = float(input("Enter The ending interval in which to search for a solution x1: "))	
    n = int(input("Enter (positive) integer The number of iterations to implement N: "))
    TOL = input("Enter the error tolerance : ")	#input for the error tolerance
    TOL = eval(TOL)
    x = Symbol('x') 
    func = lambdify(x, f, "math")
    xn = b							
    x0 = a								
    x1 = b								
    for k in range(2,n+1):																													
        xp = xn
        xn = x1 - (func(x1) * (x1 - x0)) / (func(x1) - func(x0))
        fxn=func(xn)							
        err=abs(xn-xp)/abs(xn)
        print("After " + str(k-1) + " iterations, the approximate root = " + str(xn))
        print( " f(x) = " + str(func(xn)) + " Error = " + str(err))       																																
        if ( func(xn) < TOL ):     
            break																	
        elif( err < TOL ):
            break																	
        elif ( func(x0)*func(xn) < 0 ):
            x1=xn
        else:																												
            x0 = xn
    if(err<TOL):
        print("The desired accuracy achieved; Solution converged.")
    else:
        print("The maximum number of iterations exceeded the limit.")	

def NewtonDifferenceInt2D():
    print("\n *************************************************************************** \n")
    print('Welcome to NewtonDifferenceInt 2nd Degree Method')
    print("\n *************************************************************************** \n")
    n=2
    ddf=numpy.zeros((n+1,n+1))
    f=numpy.array([5, 10, 12])
    x=numpy.array([0, 1, 3])    
    xp = int(input("Enter a real value it which the interpolation is to be obtained:: "))	
    						
    for i in range(0,n+1):
        ddf[i][0] = f[i]																						
    													
    for j in range(1,n+1):																						
    	for i in range(0,n-j+1):
    		ddf[i][j] = ( ddf[i+1][j-1] - ddf[i][j-1] ) / ( x[i+j] - x[i] )							
    						
    pro = 1
    fxp = ddf[0][0] ;																			
    for k in range(1,n+1):   
    	pro = pro * ( xp - x[k-1] )																				
    	fxp = fxp + pro * ddf[0][k]																		
    																					
    print("The interpolate or extrapolate value of function at x = xp is ",fxp)

def NewtonDifferenceIntPoly():
    print("\n *************************************************************************** \n")
    print('Welcome to NewtonDifferenceInt Degree Method')
    print("\n *************************************************************************** \n")

    n = int(input("Enter a degree of interpolating polynomial: "))
    ddf=numpy.zeros((n+1,n+1))
#    f=numpy.array((n+1))
#    x=numpy.array((n+1))

    my_array = []
    for i in range(n+1):
        print('[',i,']')
        my_array.append(int(input("Enter real values as the arbitrary nodes:")))
    x = numpy.array(my_array)
    
    my_array2 = []
    for i in range(n+1):
        print('[',i,']')
        my_array2.append(int(input(" Enter real values as the function values corresponding to x_i ")))
    f = numpy.array(my_array2)
    xp= int(input("Enter a real value it which the interpolation is to be obtained:: "))	
					
    for i in range(0,n+1):
        ddf[i][0] = f[i]																						
    													
    for j in range(1,n+1):																						
    	for i in range(0,n-j+1):
    		ddf[i][j] = ( ddf[i+1][j-1] - ddf[i][j-1] ) / ( x[i+j] - x[i] )							
    						
    pro = 1
    fxp = ddf[0][0] 																			
    for k in range(1,n+1):   
    	pro = pro * ( xp - x[k-1] )																				
    	fxp = fxp + pro * ddf[0][k]																		
    																					
    print("The interpolate or extrapolate value of function at x = xp is ",fxp)		
    


def CompositeTrapezoidal():
    print("\n *************************************************************************** \n")
    print('Welcome to Composite Trapezoidal  Method')
    print("\n *************************************************************************** \n")

#    Input Equation
    y=Equation_input()

    x = Symbol('x')     #specifying the variable used in the equation
    
    x0 = float(input("Enter the lower limit of the integral: "))
    xn = float(input("Enter the upper limit of the integral: "))
    n = int(input("Enter the number of subintervals n: "))
	
    func = lambdify(x, y, "math")		#converting the sympy function to numeric function
    h = (xn - x0)/n 
    I = func(x0) + func(xn)
    xc = x0 
    sum = 0.0 

    for j in range(1,n):
        xc = xc + h
        sum = sum + func(xc) 

    I = (h / 2.0)  *  (I + 2.0 * sum) 

    print("The approximate integral = ", I)


def CompositeSimpsons():
    print("\n *************************************************************************** \n")
    print('Welcome to Composite Simpson’s Method')
    print("\n *************************************************************************** \n")

#    Input Equation
    y=Equation_input()

    x = Symbol('x')     #specifying the variable used in the equation
    
    x0 = float(input("Enter the lower limit of the integral: "))
    xn = float(input("Enter the upper limit of the integral: "))
    n = int(input("Enter the number of subintervals n: "))
	
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
    print("The approximate integral = ", I)

def CompositeSimpsons3_8():
    print("\n *************************************************************************** \n")
    print('Welcome to Composite Simpson’s 3/8 Method')
    print("\n *************************************************************************** \n")

#    Input Equation
    y=Equation_input()

    x = Symbol('x')     #specifying the variable used in the equation
    
    x0 = float(input("Enter the lower limit of the integral: "))
    xn = float(input("Enter the upper limit of the integral: "))
    n = int(input("Enter the number of subintervals n: "))
	
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

    print("The approximate integral = ", I)
    
def JacobiMethod():
    print("\n *************************************************************************** \n")
    print('Welcome to linear system using the Jacobi method')
    print("\n *************************************************************************** \n")
    n = int(input("Enter number of unknowns: "))
    TOL = input("Enter the error tolerance : ")	#input for the error tolerance
    TOL = eval(TOL)
    print('You enter error tolerance',TOL)
    N = int(input("Enter maximum number of iterations "))
    a=numpy.zeros((n,n))
    
    print("\n ************************************************************************** \n")
    print('Enter Matrix Values of ',n,'x',n)
    for i in range(n):
        for j in range(n):
            p=('Enter Value for  row ['+str(i+1)+'] column [ '+str(j+1)+'] = ')
            a[i][j]=float(input(p))
    print(a)
    print('Enter coefficient Matrix Value')
    b=numpy.zeros(n)
    for k in range(n):
        p=('Enter coefficient Value for Matrix ['+str(k+1)+'] = ')
        b[k]=float(input(p))
        
#	 setting initial approximation as zero vector
    x=numpy.zeros(n)																																	
    for l in range(n):									
        x[l] = 0.0 	
# setting local variable
    xp=numpy.zeros(n)
    summ=0
    err=0.0																																																		
    print("The Gauss-Jacobi Method for solving a system of << ",n," >> unknowns.")																																					
#//----------------------- Processing Section -------------------------//						
    for k in range(1,N):
        for i in range (0,n):
            xp[i]=x[i]
        for i in range (0,n):
            summ=0
            for j in range (0,n):
                if (j != i):
                    summ = summ + a[i][j] * xp[j]						
            x[i] = ( b[i] - summ) / a[i][i]           																													
        summ = 0.0
#//Computing l_2-norm			
        for i in range (0,n):
            summ = summ + ( ( x[i] - xp[i] ) * ( x[i] - xp[i] ) )	
        err = sqrt(summ)																														
        if ( err < TOL ):
            break																								
#//------------------------ Output Section ----------------------------//						
					
    print(" The latest approximate solution vector is given by : " )	
    for i in range (0,n):
        print('<<',x[i],'\t')																																		
    if ( err < TOL):																									
        print("\nThe desired accuracy achieved; Solution converged.")	
    else:																													
        print("\nThe number of iterations exceeded the maximum limit.")

def GaussSeidel():
    print("\n *************************************************************************** \n")
    print('Welcome to linear system using the Gauss-Seidel')
    print("\n *************************************************************************** \n")
    n = int(input("Enter number of unknowns: "))
    TOL = input("Enter the error tolerance : ")	#input for the error tolerance
    TOL = eval(TOL)
    print('You enter error tolerance',TOL)
    N = int(input("Enter maximum number of iterations "))
    WF = float(input("Over-relaxation factor "))
   
    a=numpy.zeros((n,n))
    
    print("\n *************************************************************************** \n")
    print('Enter Matrix Values of ',n,'x',n)
    for i in range(n):
        for j in range(n):
            p=('Enter Value for  row ['+str(i+1)+'] column [ '+str(j+1)+'] = ')
            a[i][j]=float(input(p))
    print(a)
    print('Enter coefficient Matrix Value')
    b=numpy.zeros(n)
    for k in range(n):
        p=('Enter coefficient Value for Matrix ['+str(k+1)+'] = ')
        b[k]=float(input(p))
        
#	 setting initial approximation as zero vector
    x=numpy.zeros(n)																																	
    for l in range(n):									
        x[l] = 0.0 	
# setting local variable
    xp=numpy.zeros(n)
    summ=0
    err=0.0																																																		
    print("The Gauss-Seidel Method for solving a system of << ",n," >> unknowns.")																																					
#//----------------------- Processing Section -------------------------//						
    for k in range(1,N):
        for i in range (0,n):
            xp[i]=x[i]
        for i in range (0,n):
            summ=0
            for j in range (0,n):
                if (j != i):
                    summ = summ + a[i][j] * xp[j]						
            x[i] = ( b[i] - summ) / a[i][i]
            x[i] = WF * x[i] + (1 - WF) * xp[i]
         																													
        summ = 0.0
#//Computing l_2-norm			
        for i in range (0,n):
            summ = summ + ( ( x[i] - xp[i] ) * ( x[i] - xp[i] ) )	
        err = sqrt(summ)																														
        if ( err < TOL ):
            break																								
#//------------------------ Output Section ----------------------------//						
					
    print(" The latest approximate solution vector is given by : " )	
    for i in range (0,n):
        print('<<',x[i],'\t')																																		
    if ( err < TOL):																									
        print("\nThe desired accuracy achieved; Solution converged.")	
    else:																													
        print("\nThe number of iterations exceeded the maximum limit.")


def GaussianElimination():
    print("\n *************************************************************************** \n")
    print('Welcome to linear system using the near system using the Gaussian Elimination method with partial pivoting')
    print("\n *************************************************************************** \n")
    n = int(input("Enter number of unknowns: "))
    a=numpy.zeros((n,n))   
    print("\n *************************************************************************** \n")
    print('Enter Matrix Values of ',n,'x',n)
    for i in range(n):
        for j in range(n):
            p=('Enter Value for  row ['+str(i+1)+'] column [ '+str(j+1)+'] = ')
            a[i][j]=float(input(p))
    print(a)
    print('Enter coefficient Matrix Value')
    b=numpy.zeros(n)
    for k in range(n):
        p=('Enter coefficient Value for Matrix ['+str(k+1)+'] = ')
        b[k]=float(input(p))
        
#	 setting initial approximation as zero vector
    x=numpy.zeros(n)																																	
    for l in range(n):									
        x[l] = 0.0 	
																																																	
    print("The near system using the Gaussian Elimination method with partial pivoting for solving a system of << ",n," >> unknowns.")																																					
#//----------------------- Processing Section -------------------------//						
#// Forward Elimination Phase																						
#// Searching largest absolute coefficient in the ith column for partial pivoting	
    for i in range (0,n-1):
        r=i
        for j in range (i+1,n):   
            if ( abs( a[r][i] )<abs( a[j][i] ) ):
                r=j
        if (a[r][i] == 0 ):
            print( "The given system has no unique solution" )										
            break
        elif( r != i ):
            for j in range (0,n):
                temp = a[i][j]																							
                a[i][j] = a[r][j]																						
                a[r][j] = temp
        temp1 = b[i]																								
        b[i] = b[r]																										
        b[r] = temp1
#//row replacement in the augmented matrix for eliminating the coefficient below the pivot	
        for k in range(i+1,n):
            multiplier = a[k][i] / a[i][i]
            for j in range (i+1,n):
                a[k][j] = a[k][j] - multiplier * a[i][j]
            b[k] = b[k] - multiplier * b[i] 	
    																																		
    if ( a[n-1][n-1]==0 ):																					
        print("The system has no unique solution. " )
#        terminates the program immediately
        sys.exit()										
    else:																														
        x[n-1] = b[n-1] / a[n-1][n-1]																	
    for i in range(n-2,-1,-1):																																																																						
        summ = 0.0 	
        for j in range (i+1,n):												
            summ = summ + a[i][j] * x[j]																														
        x[i] = ( b[i] - summ ) / a[i][i]	
	#//------------------------ Output Section ----------------------------//																																							
    print("The solution of the given system is ")
    r=[]
    for i in range (0,n):
#        print (x[i])
        r.append(x[i])
    print('X = ',r)



def Doolittle():
    print("\n *************************************************************************** \n")
    print('Welcome to linear system using the near system using the Doolittle’s method ')
    print("\n *************************************************************************** \n")
    n = int(input("Enter number of unknowns: "))
    a=numpy.zeros((n,n))   
    print("\n *************************************************************************** \n")
    print('Enter Matrix Values of ',n,'x',n)
    for i in range(n):
        for j in range(n):
            p=('Enter Value for  row ['+str(i+1)+'] column [ '+str(j+1)+'] = ')
            a[i][j]=float(input(p))
    print(a)
    print('Enter coefficient Matrix Value')
    b=numpy.zeros(n)
    for k in range(n):
        p=('Enter coefficient Value for Matrix ['+str(k+1)+'] = ')
        b[k]=float(input(p))
        
#	 setting initial approximation as zero vector
    x=numpy.zeros(n)																																	
    for l in range(n):									
        x[l] = 0.0 	
# setting local variable
    u=numpy.zeros((n,n))   
    l=numpy.zeros((n,n))
    y=numpy.zeros(n)																																	
																																	
#//---------------------- Processing Section ----------------------//							
#   // Formation of L and U as factors of A, i.e., A=LU							
    for i in range (0,n):
            for j in range (0,n):
                l[i][j] = u[i][j] = 0.0
    for j in range (0,n):
        u[0][j] = a[0][j]																							
        l[j][0] = a[j][0] / u[0][0]
    for i in range (1,n):
        l[i][i] = 1
        for j in range (i,n):
            summ=0.0
            for s in range (0,i):
                summ = summ + l[i][s] * u[s][j]					
            u[i][j] = a[i][j] - summ 
            
        for j in range (i+1,n):
            summ = 0.0 
            for s in range (0,i):							
                summ = summ + l[j][s] * u[s][i] 						
            l[j][i] = ( a[j][i] - summ) / u[i][i]
                																																																															
#    // Forward substitution phase for solving LY=B							
																																		
    y[0] = b[0]
    for i in range (1,n):																			
        summ = 0.0
        for j in range (0,i):										
            summ = summ + l[i][j] * y[j] 							
        y[i] = b[i] - summ 											
																																
#    // Back Substitution Phase for solving UX=Y							
																																		
    x[n-1] = y[n-1] / u[n-1][n-1]
    for i in range (n-2,-1,-1):																																					
        summ = 0.0
        for j in range (i+1,n):																					
            summ = summ + u[i][j] * x[j]							
        x[i] = ( y[i] - summ ) /  u[i][i] 							
																																	
#    //---------------------- Output Section ----------------------//							
																																		
    print("The L matrix is" )
    for i in range (0,n):
        for j in range (0,n):																														
            print(l[i][j], " \t " )	
																		
    print("The U matrix is" )
    for i in range (0,n):
        for j in range (0,n):
            print(u[i][j], " \t " )																						
																
    print("The required solution is" )
    for i in range (0,n):																														
        print(x[i], " \t " )

def Crouts():
    print("\n *************************************************************************** \n")
    print('Welcome to linear system using the near system using the Crouts method ')
    print("\n *************************************************************************** \n")
    n = int(input("Enter number of unknowns: "))
    a=numpy.zeros((n,n))   
    print("\n *************************************************************************** \n")
    print('Enter Matrix Values of ',n,'x',n)
    for i in range(n):
        for j in range(n):
            p=('Enter Value for row ['+str(i+1)+'] column [ '+str(j+1)+'] = ')
            a[i][j]=float(input(p))
    print(a)
    print('Enter coefficient Matrix Value')
    b=numpy.zeros(n)
    for k in range(n):
        p=('Enter coefficient Value for Matrix ['+str(k+1)+'] = ')
        b[k]=float(input(p))
        
#	 setting initial approximation as zero vector
    x=numpy.zeros(n)																																	
    for l in range(n):									
        x[l] = 0.0 	
# setting local variable
    u=numpy.zeros((n,n))   
    l=numpy.zeros((n,n))
    y=numpy.zeros(n)																																	
																																	
#//---------------------- Processing Section ----------------------//							
#   // Formation of L and U as factors of A, i.e., A=LU							
    for i in range (0,n):
            for j in range (0,n):
                l[i][j] = u[i][j] = 0.0
    for j in range (0,n):
        l[j][0] = a[j][0]																						
        u[0][j] = a[0][j] / l[0][0]
        
    for i in range (1,n):
        u[i][i] = 1
        for j in range (i,n):
            summ=0.0
            for s in range (0,i):
                summ = summ + l[j][s] * u[s][i]						
            l[j][i] = a[j][i] - summ

        for j in range (i+1,n):
            summ = 0.0 
            for s in range (0,i):	
                summ = summ + l[i][s] * u[s][j] 					
            u[i][j] = ( a[i][j] - summ ) / l[i][i] 
 																																																															
#    // Forward substitution phase for solving LY=B							
    y[0] = b[0] / l[0][0]																																																								
    for i in range (1,n):																			
        summ = 0.0
        for j in range (0,i):
            summ = summ + l[i][j] * y[j] 							
        y[i] = ( b[i] - summ ) / l[i][i] 
																														
#    // Back Substitution Phase for solving UX=Y							
    x[n-1] = y[n-1] 																					
    for i in range (n-2,-1,-1):																																					
        summ = 0.0
        for j in range (i+1,n):																					
            summ = summ + u[i][j] * x[j]							
        x[i] = y[i] - summ 		
					
																																	
#    //---------------------- Output Section ----------------------//							
																																		
    print("The L matrix is" )
    for i in range (0,n):
        for j in range (0,n):																														
            print(l[i][j], " \t " )	
																		
    print("The U matrix is" )
    for i in range (0,n):
        for j in range (0,n):
            print(u[i][j], " \t " )																						
																
    print("The required solution is" )
    for i in range (0,n):																														
        print(x[i], " \t " )			

def Cholesky():
    print("\n *************************************************************************** \n")
    print('Welcome to linear system using the near system using the Cholesky’s  method ')
    print("\n *************************************************************************** \n")
    n = int(input("Enter number of unknowns: "))
    a=numpy.zeros((n,n))   
    print("\n *************************************************************************** \n")
    print('Enter Matrix Values of ',n,'x',n)
    for i in range(n):
        for j in range(n):
            p=('Enter Value for row ['+str(i+1)+'] column [ '+str(j)+'] = ')
            a[i][j]=float(input(p))
    print(a)
    print('Enter coefficient Matrix Value')
    b=numpy.zeros(n)
    for k in range(n):
        p=('Enter coefficient Value for Matrix ['+str(k+1)+'] = ')
        b[k]=float(input(p))
        
#	 setting initial approximation as zero vector
    x=numpy.zeros(n)																																	
    for l in range(n):									
        x[l] = 0.0 	
# setting local variable
    l=numpy.zeros((n,n))
    y=numpy.zeros(n)																																	
																																	
#//---------------------- Processing Section ----------------------//							
#   // Formation of L and U as factors of A, i.e., A=LU							
    for i in range (0,n):
            for j in range (0,n):
                l[i][j] = 0.0
    l[0][0] = sqrt( a[0][0] ) 
                
    for j in range (1,n):
        l[j][0] = a[j][0] / l[0][0]																																										
        
    for i in range (1,n):
        summ = 0.0
        for k in range (0,i):									
            summ = summ + l[i][k] * l[i][k]
        comp = cmath.sqrt( a[i][i] - summ )
        l[i][i] = comp.real

        for j in range (i+1,n):
            summ = 0.0 
            for k in range (0,i):
                summ = summ + l[i][k] * l[j][k] 					
            l[j][i] = (a[j][i] - summ ) / l[i][i]
																																																															
#    // Forward substitution phase for solving LY=B							
    y[0] = b[0] / l[0][0]																																																								
    for i in range (1,n):																			
        summ = 0.0
        for j in range (0,i):
            summ = summ + l[i][j] * y[j] 							
        y[i] = ( b[i] - summ ) / l[i][i] 
																														
#    // Back Substitution Phase for solving UX=Y							
    x[n-1] = y[n-1] / l[n-1][n-1]																				
    for i in range (n-2,-1,-1):																																					
        summ = 0.0
        for j in range (i+1,n):
            summ = summ + l[j][i] * x[j]						
        x[i] = (y[i] - summ ) / l[i][i]
																																	
#    //---------------------- Output Section ----------------------//							
																																		
    print("The L matrix is" )
    for i in range (0,n):
        for j in range (0,n):																														
            print(l[i][j], " \t " )																																							
																
    print("The required solution is" )
    for i in range (0,n):																														
        print(x[i], " \t " )																																								

def PowerMethod():
    print("\n *************************************************************************** \n")
    print('Welcome to ldominant eigenvalue of the  matrix using the Power method')
    print("\n *************************************************************************** \n")
    n = int(input("Enter number of unknowns: "))
    TOL = input("Enter the error tolerance : ")	#input for the error tolerance
    TOL = eval(TOL)
    print('You enter error tolerance',TOL)
    N = int(input("Enter maximum number of iterations "))
    a=numpy.zeros((n,n))
   
    a=numpy.zeros((n,n))   
    print("\n *************************************************************************** \n")
    print('Enter Matrix Values of ',n,'x',n)
    for i in range(n):
        for j in range(n):
            p=('Enter Value for row ['+str(i+1)+'] column [ '+str(j)+'] = ')
            a[i][j]=float(input(p))
    print(a)
        
#	initial approx. to the dominant eigenvector
    x=numpy.zeros(n)																																	
    for l in range(n):									
        x[l] = 1 	
# setting local variable
    xp=numpy.zeros(n)																																	
																																	
#//---------------------- Processing Section ----------------------//													
    for k in range (1,N):
        for i in range (0,n):
            xp[i] = x[i]
#// Computing the vector X^(k) = A * X^(k-1)					
        for i in range (0,n):																				
            summ = 0.0
            for j in range (0,n):												
                summ = summ + a[i][j] * xp[j]						
            x[i] = summ
#        // Approximating the eigenvalue B and normalizing the vector X					
        r = 0
        for i in range (1,n):
            if ( abs(x[i]) > abs(x[r]) ):
                r = i 																																								
        B = x[r]
        for i in range (0,n):																		
            x[i] = x[i] / B 																																																							
#      // Computing the error as L2-norm								
        sum1 = 0.0
        for i in range (0,n):														
            sum1 = sum1 + ( x[i] - xp[i] ) * ( x[i] - xp[i] )		
        err = sqrt( sum1 )
        if ( err < TOL ):
            break																																	
#    //---------------------- Output Section ----------------------//							
																																		
    print("\n The approximate dominant eigenvalue is ", B, "\n The approximate corresponding eigenvector is " )
    for i in range (0,n):																														
        print(x[i], " \t " )
    if( err<TOL ):																								
        print("\nThe desired accuracy achieved; Solution converged")				
    else:																														
        print("\nThe number of iterations exceeded the maximum limit.")
	