#!/usr/bin/python
import tempfile
import pexpect

def ssh(cmd,copy=0, host="cseproj146.cse.iitk.ac.in", user="siddsax", password="welcome123", timeout=300, bg_run=False,dst=None):                           
    """SSH'es to a host using the supplied credentials and executes a command.                           
    Throws an exception if the command doesn't return 0.              
    bgrun: run command in the background"""          
    fname = tempfile.mktemp()
    fout = open(fname, 'w')
    options = '-q -oStrictHostKeyChecking=no -oUserKnownHostsFile=/dev/null -oPubkeyAuthentication=no'
    if bg_run:
        options += ' -f'
    if(copy):
        origin = cmd
        ssh_cmd = 'scp %s %s@%s:%s' % (origin, user, host, dst)        
    else:           
        ssh_cmd = 'ssh %s@%s %s "%s"' % (user, host, options, cmd)        
    child = pexpect.spawn(ssh_cmd, timeout=timeout)                   
    child.expect(['password: '])       
    child.sendline(password)       
    child.logfile = fout           
    child.expect(pexpect.EOF)      
    child.close()                  
    fout.close()                   

    fin = open(fname, 'r')         
    stdout = fin.read()            
    fin.close()                    

    if 0 != child.exitstatus:      
        raise Exception(stdout)    

    return stdout
