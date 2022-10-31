import os
import time
if __name__ == '__main__':
    all_cls = ["spot_bin", "spot_mul"]
    all_embed_model = ["gru", "dpcnn"]
    # all_hybrid = ["3loss_src", "3loss_ir", "3loss_byte1", "3loss_byte2", "3loss_src_ir"]
    all_hybrid = ["3loss_src", "3loss_ir"]
    # all_hybrid = ["3loss_byte1", "3loss_byte2"]

    gpu = 2
    gpu_num = 4
    hidden = 128
    layers = 4
    for hybrid in all_hybrid:
        for cls in all_cls:
            for embed in all_embed_model:
                if embed == "gru":
                    classifier = "gru"
                elif embed == "dpcnn":
                    classifier = "lr"
                if "byte1" in hybrid:
                    byte_name = "byte_id_1"
                elif "byte2" in hybrid:
                    byte_name = "byte_id_2"
                else:
                    byte_name = "byte_id_1"
                if hybrid == "3loss_src" and cls == "spot_bin":
                    epoch = 60
                elif hybrid == "3loss_src" and cls == "spot_mul":
                    epoch = 80
                elif hybrid == "3loss_ir" and cls == "spot_bin":
                    epoch = 60
                elif hybrid == "3loss_ir" and cls == "spot_mul":
                    epoch = 100
                elif hybrid == "3loss_byte1" or hybrid == "3loss_byte2" and cls == "spot_bin":
                    epoch = 90
                elif hybrid == "3loss_byte1" or hybrid == "3loss_byte2" and cls == "spot_mul":
                    epoch = 100
                name = hybrid+"-"+cls+"-"+embed
                logfile = hybrid+"_w2v_"+embed+"_"+classifier+"_"+cls+".log"
                cmd = "tmux new-window -n "+name+" \"CUDA_VISIBLE_DEVICES="+str(gpu)+" python -u entry.py -e1 w2v -e2 src -i1 w2v -i2 ir_id_1 -b1 w2v -b2 "+byte_name+" -emb "+embed+" -r False -s True -d 3loss -l "+cls+" -c "+hybrid+" -n "+classifier+" -epo "+str(epoch)+" -hid "+str(hidden)+" -lay "+str(layers)+" > errlog/"+logfile+" 2>&1\""
                temp = "python -u entry.py -e1 w2v -e2 src -i1 w2v -i2 ir_id_1 -b1 w2v -b2 "+byte_name+" -emb "+embed+" -r False -s True -d 3loss -l "+cls+" -c "+hybrid+" -n gru -epo "+str(epoch)+" -hid "+str(hidden)+" -lay "+str(layers)
                cmd2 = "ps -ef|grep \""+temp+"\" |grep -v grep|wc -l"
                print("start running "+hybrid+ " "+cls+" "+embed)
                print(cmd)
                os.system(cmd)
                time.sleep(20)
                f = os.popen(cmd2, "r")
                d = f.read()
                if int(d)!=0:
                    print("start successfully!")
                elif int(d)==0:
                    print("fail to start")
        gpu += 1





