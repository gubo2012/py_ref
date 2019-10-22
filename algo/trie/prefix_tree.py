# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 21:45:36 2019

@author: GuBo
"""

class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.trie = {}
        

    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        t = self.trie
        for c in word:
            if c not in t:
                t[c] = {}
            t = t[c]
        t['.'] = {} # end

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        t = self.trie
        i = 0
        failed = False
        while i < len(word) and not failed:
            failed = word[i] not in t
            if not failed:
                t = t[word[i]]
                i += 1
        return '.' in t and not failed
    

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        t = self.trie
        i = 0
        failed = False
        while i < len(prefix) and not failed:
            failed = prefix[i] not in t
            if not failed:
                t = t[prefix[i]]
                i += 1        
        return not failed
        


# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)
        
trie = Trie()

trie.insert("apple");
print(trie.search("apple"))   # returns true
print(trie.search("applee"))
print(trie.search("app"))     # returns false
print(trie.startsWith("app")); # returns true
trie.insert("app");   
print(trie.search("app"));     # returns true