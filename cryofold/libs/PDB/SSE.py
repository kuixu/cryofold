import numpy as np


def get_sse_records_by_class(sse_dict):

    chainids  = np.array(sse_dict['cha'])
    c_res_ids = np.array(sse_dict['id'])
    c_res_names = np.array(sse_dict['res'])
    c_sses    = np.array(sse_dict['sse'])

    ch_ids    = np.unique(chainids)

    
    # import pdb; pdb.set_trace()
    str_h = ""
    str_s = ""
    for ch_id in ch_ids:
        ind = chainids==ch_id
        chain = chainids[ind]
        res_ids = c_res_ids[ind]
        res_names = c_res_names[ind]
        sse     = c_sses[ind]


        
        begtmp = -1
        beg = 0
        endtmp = -1
        end = -1
        h_id = 0
        is_write = False




        # chain, res_ids, res_names, sse
        # np.unique(np.array(chain))

        for i in range(len(sse)):
            if sse[i] == 1:
                if i == 0 or sse[i - 1] != 1: 
                    beg = i

                if i == len(sse) -1 or sse[i + 1] != 1:
                    end = i
                    is_write = True
            # elif i == 0 and sse[i] != 1:
            #     continue
            # elif sse[i - 1] != 1 and sse[i] == 1:
            #     beg = i
            # if sse[i] == 1 :
            #     if i == len(sse) -1 or sse[i + 1] != 1:
            #         end = i

            # if beg != begtmp and end != endtmp:
            if beg <= end and is_write:
                is_write = False
                # print(i,beg,end,'\n')
                begtmp = beg
                endtmp = end
                h_id = h_id + 1

                helixID = "%3s" % (h_id)
                beg_label = "%-3s" % res_names[beg]
                end_label = "%-3s" % res_names[end]
                beg_chain = "%2s" % chain[i]
                end_chain = "%2s" % chain[i]
                beg_seqid = "%4d" % int(res_ids[beg])
                end_seqid = "%4d" % int(res_ids[end])
                initICode = "%s" % (" ")
                endICode = "%s" % (" ")
                helixClass = "%2d" % 1
                comment = "%-30s" % (" ")
                length = "%5d" % (res_ids[end] - res_ids[beg] + 1)
                str_h = (
                    str_h
                    + "%-6s" % ("HELIX")
                    + " "
                    + helixID
                    + " "
                    + helixID
                    + " "
                    + beg_label
                    + ""
                    + beg_chain
                    + " "
                    + beg_seqid
                    + initICode
                    + " "
                    + end_label
                    + ""
                    + end_chain
                    + " "
                    + end_seqid
                    + endICode
                    + helixClass
                    + comment
                    + " "
                    + length
                    + "\n"
                )

        
        begtmp = -1
        beg = -1
        endtmp = -1
        end = -1
        s_id = 0
        is_write = False
        for i in range(len(sse)):
            # if i == 0 and sse[i] == 2:
            #     beg = i
            # elif i == 0 and sse[i] != 2:
            #     continue
            # elif sse[i - 1] != 2 and sse[i] == 2:
            #     beg = i
            # if sse[i] == 2 :
            #     if i == len(sse) - 1 or sse[i + 1] != 2:
            #         end = i
            if sse[i] == 2:
                if i == 0 or sse[i - 1] != 2: 
                    beg = i

                if i == len(sse) -1 or sse[i + 1] != 2:
                    end = i
                    is_write=True
            if beg>=0 and beg <= end and is_write:
                is_write = False
            # if beg != begtmp and end != endtmp and end-beg>0:
                # print(i,beg,end,'\n')
                begtmp = beg
                endtmp = end
                s_id = s_id + 1

                strand = "%3d" % (s_id)
                sheetID = "%3s" % (s_id)
                numStrands = "%2s" % (1)
                beg_label = "%-3s" % (res_names[beg])
                end_label = "%-3s" % (res_names[end])
                beg_chain = "%2s" % (chain[i])
                end_chain = "%2s" % (chain[i])
                beg_seqid = "%4d" % (int(res_ids[beg]))
                end_seqid = "%4d" % (int(res_ids[end]))
                initICode = "%s" % (" ")
                endICode = "%s" % (" ")
                sense = 0
                sense = "%2d" % (sense)
                curAtom = ""
                curResName = ""
                curChainId = ""
                curResSeq = ""
                curICode = ""

                preAtom = ""
                preResName = ""
                preChainId = ""
                preResSeq = ""
                preICode = ""
                str_s = (
                    str_s
                    + "%-6s" % ("SHEET")
                    + " "
                    + strand
                    + " "
                    + sheetID
                    + numStrands
                    + " "
                    + beg_label
                    + ""
                    + beg_chain
                    + beg_seqid
                    + initICode
                    + " "
                    + end_label
                    + ""
                    + end_chain
                    + end_seqid
                    + endICode
                    + sense
                    + " "
                    + curAtom
                    + curResName
                    + " "
                    + curChainId
                    + curResSeq
                    + curICode
                    + " "
                    + preAtom
                    + preResName
                    + " "
                    + preChainId
                    + preResSeq
                    + preICode
                    + "\n"
                )
    # import pdb;pdb.set_trace()
    return str_h + str_s