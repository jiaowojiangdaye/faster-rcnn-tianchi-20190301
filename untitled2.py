#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 20:39:58 2019
1 5 8

@author: mosay
"""
def task0():
    #x, y, m = [int(x) for x in input().strip().split(' ')]
    ##x,y,m = 2,3,15
    #
    #if x<=0 and y<=0 and max(x,y) < m:
    #    print(-1)
    #elif x == m or y == m: 
    #    print(0)
    #else:
    #    iters = int((m-min(x,y))/max(x, y))
    #    if (m-min(x,y))%max(x, y) != 0:
    #        iters+= 1
    #
    #    print(iters)
    
    
    x,y,m = map(int,input().split())
    #x,y,m = 2,15,15
    
    if x<=0 and y<=0 and max(x,y) < m:
        print(-1)
    else:
        if max(x, y) >= m:
            print(0)
        else:
            temp = x + y
            if x < y:
                x = temp
            else:
                y = temp
            iter = 1
            while x < m:
                iter += 1
                y += x
                x, y = y, x
    
            print(iter)

def task1():
    n = int(input())
    #n = 2
    if n ==0:
        print(0)
    else:
        xx = []
        aa = []
        for i in range(n):
            a, b = [int(x) for x in input().strip().split(' ')]
            xx.append(a)
            aa.append(b)
    #    n = 2
    #    xx = [-6, -3]
    #    aa = [2, 4]
        
        xx_zip = zip(xx, aa)
        xx_sorted = sorted(xx_zip, key=lambda x:x[0])
        
        
        mid_idx = 0
        poistive = False
        for idx, one in enumerate(xx_sorted):
            if one[0] > 0:
                mid_idx = idx
                poistive = True
                break
        if not poistive:
            print(xx_sorted[-1][1])
        else:
            left_num, right_num = mid_idx, n-mid_idx        
            
            if abs(left_num-right_num)<=1:
                print(sum(aa))
            else:
                if left_num > right_num:
                    sum = 0
                    for one in xx_sorted[mid_idx-right_num-1:]:
                        sum += one[1]
                    print(sum)
                else:
                    sum = 0
                    for one in xx_sorted[:mid_idx*2+1]:
                        sum += one[1]
                    print(sum)



def task_20190322():
    
    An = [0, 1,2,3,3,5,5,5,6,8,8,9,10, 11,12,18]
    s = 10
    
    i = 0
    j = len(An)-1
    counter = 0
    while 0<=i and i<j and j<=len(An):
        temp = An[i]+An[j]
        if temp > s:
            j -= 1
        elif temp < s:
            i += 1
        else:
            ii = i
            jj = j
            while An[ii+1] == An[ii]:
                jj = j
                while jj > ii and An[jj-1] == An[jj]:
                    print('A{}({})+A{}({})={}'.format(ii,An[ii],jj, An[jj], temp))
                    counter += 1
                    jj -= 1
                ii += 1
                
            if ii > i:
                i = ii+1
            if jj<j:
                j = jj-1
            if ii == i and jj == j:
                i += 1
    print(counter)


def task_20190322_1():
    
    An = [1,5,5,5,5,5,8,9,9]
    s = 10
    
    i = 0
    j = len(An)-1
    counter = 0
    
    while 0<=i and i<j and j<=len(An):
        temp = An[i]+An[j]
        if temp > s:
            j -= 1
        elif temp < s:
            i += 1
        else:
            left_count = 0
            right_count = 0
            if An[i] == An[j]:
                counter+=int((j-i+1)*(j-i)/2)
                return counter
            else:
                while i<=j and An[i]+An[j] == s:
                    i+=1
                    left_count += 1
                i -= 1
                
                while An[i]+An[j] == s:
                    j-=1
                    right_count +=1
                i += 1
                counter+=left_count*right_count
        
                
    return counter

       
print(task_20190322())




