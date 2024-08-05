
import gzip
from typing import Sequence, Tuple, List

def bi_search(item, sorted_list, retern_index=False):
    """
    pattern             -- Regex expression
    sorted_list         -- A increasingly sorted list
    retern_index        -- If retern_index==True, the index will be returned

    Return:
        if retern_index == False
            True if item in sorted_list
            False if item not in sorted_list
        else
            sorted_list.index(item)
    """
    start = 0
    end = len(sorted_list) - 1

    while start <= end:
        middle = (start + end) // 2
        if sorted_list[middle] < item:
            start = middle + 1

        elif sorted_list[middle] > item:
            end = middle - 1

        else:
            return middle if retern_index else True

    return -1 if retern_index else False

def read_ffindex(ffindex_file: str) -> Tuple[Sequence, Sequence, Sequence]:
    key_list = []
    start_list = []
    size_list = []
    for line in open(ffindex_file):
        key, start, size = line.strip().split()
        start, size = int(start), int(size)
        key_list.append(key)
        start_list.append(start)
        size_list.append(size)
    ## Check key is sorted
    for i in range(len(key_list)-1):
        assert key_list[i] < key_list[i+1], "Unsorted key"
    return key_list, start_list, size_list

# class FFindex:
#     """
#     Read FFindex file:
#         ffindex = FFindex('test.ffdata', 'test.ffindex')
#         ff = ffindex.get("103L.cif.gz")
#     """
#     def __init__(self, ffdata_file: str, ffindex_file: str):
#         self.ffdata_file = ffdata_file
#         self.ffindex_file = ffindex_file
#         self.key_list, self.start_list, \
#             self.size_list = read_ffindex(ffindex_file)
#         self.ffdata = open(ffdata_file, 'rb')

#     def get(self, key: str, decompress: bool = True, decode: bool = True) -> str:
#         index = bi_search(key, self.key_list, True)
#         assert index != -1, f"Error: {key} not found in FFdata"
#         start, size = self.start_list[index], self.size_list[index]
#         self.ffdata.seek(start)
#         content = self.ffdata.read(size)
#         if decompress:
#             import gzip
#             content = gzip.decompress(content)
#         if decode:
#             content = content.decode()
#         return content

#     def has(self, key: str) -> bool:
#         index = bi_search(key, self.key_list, True)
#         return True if index != -1 else False
class FFindex:
    """
    Read FFindex file:
        ffindex = FFindex('test.ffdata', 'test.ffindex')
        ff = ffindex.get("103L.cif.gz")
    """
    def __init__(self, ffdata_file: str, ffindex_file: str, dynamic_file_handle: bool = False):
        self.ffdata_file = ffdata_file
        self.ffindex_file = ffindex_file
        self.key_list, self.start_list, \
            self.size_list = read_ffindex(ffindex_file)
        self.dynamic_file_handle = dynamic_file_handle
        if not self.dynamic_file_handle:
            self.ffdata = open(ffdata_file, 'rb')
    
    def get(self, key: str, decompress: bool = True, decode: bool = True) -> str:
        index = bi_search(key, self.key_list, True)
        assert index != -1, f"Error: {key} not found in FFdata"
        start, size = self.start_list[index], self.size_list[index]
        if self.dynamic_file_handle:
            ffdata = open(self.ffdata_file, 'rb')
            ffdata.seek(start)
            content = ffdata.read(size)
            ffdata.close()
        else:
            self.ffdata.seek(start)
            content = self.ffdata.read(size)
        if decompress:
            import gzip
            content = gzip.decompress(content)
        if decode:
            content = content.decode()
        return content
    
    def has(self, key: str) -> bool:
        index = bi_search(key, self.key_list, True)
        return True if index != -1 else False
