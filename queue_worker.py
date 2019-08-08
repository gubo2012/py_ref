# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 18:45:01 2019

@author: dxuser22
"""

import queue
from threading import Thread
import random
import time

num_threads = 2


def demo_func(args):
    print(args[0], args[1])

def start_worker(q, worker_num, func):
    
    while not q.empty():
        task = q.get()
        if task is None:
            break
        
        print('start task {} by worker # {}'.format(task[0], worker_num))
        
        func(task[1])       
              
        time.sleep(1 + random.random())
        q.task_done()
    
    print('no more tasks')


def start_pool(q, num_threads, func):
    for i in range(num_threads):
        worker = Thread(target = start_worker, args = (q, i, func))
        worker.setDaemon(True)
        worker.start()
    


if __name__ == '__main__':
    
    t1 = time.time()
    q = queue.Queue()
    
    demo = 1
    
    if demo == 1:

    # demo 1    
    #    fill the queue with work tasks
        for i in range(30):
            q.put((i, ('abc', i*2, i*i)))
            
        func = demo_func    
        
        start_pool(q, num_threads, func)
    
    else:
        # demo 2, real case        
        from comb_events import comb_events_main_process
        customers = ['kroger', 'target', 'meijerSun', 'meijerThu']
        func = comb_events_main_process
        for i in range(len(customers)):
            q.put((i, customers[i]))
            
        start_pool(q, num_threads, func)
        
q.join()
print('________')
print('all tasks are done')
t2 = time.time() - t1
print('ran in: {} sec'.format(t2))       

        
    