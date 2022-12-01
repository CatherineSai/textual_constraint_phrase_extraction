from functools import reduce
from itertools import groupby
from spacy.tokens import Span, Doc
from spacy import displacy

# annotation labels of spans
categories = ['SUBJECT', 'SIGNAL', 'VERB', 'TIME', 'CONDITION', 'OBJECT', 'OP_SUBJECT', 'OP_SIGNAL', 'OP_VERB', 'OP_TIME', 'OP_CONDITION', 'OP_OBJECT']

# defined trigger words in addition to dependency tags

signal = ['should', 'shall', 'must', 'may']

condition_trigger = ['without', 'within', 'where not', 'where', 'when not', 'when', 'upon', 'until not', 'until', 'unless not', 'unless and until', 'unless', 'timely', 'taking into account', 'subject to', 'regardless of', 'provided that not', 'provided that', 'prior to', 'only if', 'not to exceed', 'not subject to', 'not later than', 'not equal to', 'not earlier than' 'no later than', 'not earlier than', 'no more than', 'no less than', 'no later than', 'no earlier than', 'more than or equal to', 'more than', 'minimum of', 'minimum', 'maximum of', 'maximum', 'lesser than', 'lesser of', 'lesser', 'less than or equal to', 'less than', 'least of', 'least', 'later than', 'last of', 'irrespective of', 'in the case of', 'in the absence of', 'if not', 'if', 'highest', 'greatest of', 'greater than or equal to', 'greater than', 'greater of', 'greater', 'first of', 'extended', 'expressly', 'except', 'exceeds', 'exceed', 'exactly', 'equal to', 'earlier than', 'during', 'conditioned upon', 'conditioned on', 'before', 'at the time when', 'at the time', 'at the latest', 'at most', 'at least', 'as soon as', 'as long as', 'after']

time_point = ['years', 'year', 'weeks', 'week', 'seconds', 'second', 'period', 'periods', 'months', 'month', 'minutes', 'minute', 'hours', 'hour', 'days', 'day']

relative_words = ['which', 'who', 'that', 'whose', 'whom', 'where', 'what']




# Phrase is a list of tokens that would represent e.g. a subject phrase, etc.
class Phrase:
    def __init__(self, tokens_or_root=[], has_skips=False):
        self.has_skips = has_skips # this phrase is not guaranteed to be fully gapless in the sentence

        if type(tokens_or_root) is list: # list of tokens
            self.root = None
            self.tokens = tokens_or_root
        else: # a root that we would expand
            self.root = tokens_or_root
            self.tokens = expand_subtree(self.root)

    def as_span(self, label):
        '''
        return a list of Span objects and their labels, such that each gapless index sequences are mapped to one span object e.g. [[3:7], [10:11], [15:17]]
        '''
        indices = list(set(t.i for t in self.tokens))
        indices.sort()
        spans = []
        # function found on stackoverflow https://stackoverflow.com/questions/4628333/converting-a-list-of-integers-into-range-in-python
        for group_id, group in groupby(enumerate(indices), lambda pair: pair[1] - pair[0]):
            group = list(group)
            spans.append(Span(self.tokens[0].doc, group[0][1], group[-1][1]+1, label))
        return spans

    def as_str(self, lower=True):
        text = ' '.join([t.text for t in self.tokens]) if self.has_skips else ''.join([t.text_with_ws for t in self.tokens])
        return text.lower() if lower else text
    
    def merge_as_str(phrases):
        if phrases is None:
            return ''
        return ' | '.join([str(phrase) for phrase in phrases])

    def merge(phrases):
        if phrases is None:
            return Phrase([])
        return reduce(lambda x, y: x+y, phrases, Phrase([]))

    def get_children(self, as_set=False):
        '''children of a phrase are all of their individual children minus their own inner chunk'''
        all_children = set()
        for token in self.tokens:
            all_children |= set(token.children)
        
        all_children -= set(self.tokens)
        if not as_set:
            all_children = list(all_children)
            all_children.sort(key=by_index)
        return all_children

    def has_trigger(self, triggers):
        string = self.as_str()
        for t in triggers:
            # check if it's in the string
            if string.find(t) != -1:
                return True
        return False

    def starts_with_trigger(self, triggers):
        string = self.as_str()
        for t in triggers:
            # check if it's at the front of the string
            if string.find(t) == 0:
                return True
        return False
    
    def __getitem__(self, key):
        return self.tokens[key]
    
    def __len__(self):
        return len(self.tokens)
    
    def __str__(self):
        return self.as_str(lower=False)
    
    def __add__(self, other):
        return Phrase(self.tokens + other.tokens if other is not None else [], has_skips=True)
    
    def __sub__(self, other):
        if type(other) is Phrase:
            blacklist = set(other.tokens)
            return Phrase([t for t in self.tokens if t not in blacklist], has_skips=True)
        else: #type(other) is Token:
            return Phrase([t for t in self.tokens if t != other], has_skips=True)


# Extracted contains the extracted subject phrases, verb phrases, etc.
class Extracted:
    def __init__(self, whole_phrase, subject_phrases=None, signal_word=None, verb_phrase=None, times=None, conditions=None, objects=None):
        '''
        whole_phrase, subject_phrases, times, conditions: [Phrase];
        signal_word, verb_phrase: Phrase;
        objects: [Extracted] OR [Phrase]
        '''
        self.whole_phrase = whole_phrase
        self.subject_phrases = subject_phrases
        self.signal_word = signal_word
        self.verb_phrase = verb_phrase
        self.times = times
        self.conditions = conditions
        self.objects = objects

    def as_tuple(self):
        return (self.whole_phrase, self.subject_phrases, self.signal_word, self.verb_phrase, self.times, self.conditions, self.objects)

    def as_row(self):
        def str_or_empty(p):
            return str(p) if p is not None else ''

        wp = str_or_empty(self.whole_phrase)
        sp = Phrase.merge_as_str(self.subject_phrases)
        sw = str_or_empty(self.signal_word)
        vp = str_or_empty(self.verb_phrase)
        t = Phrase.merge_as_str(self.times)
        c = Phrase.merge_as_str(self.conditions)
        
        op, op_subj, op_sig, op_verb, op_time, op_cond, op_obj = [], [], [], [], [], [], []
        for object in self.objects:
            if type(object) is Extracted:
                obj = object.as_tuple()
                op.append(str_or_empty(obj[0]))
                op_subj.append(Phrase.merge_as_str(obj[1]))
                op_sig.append(str_or_empty(obj[2]))
                op_verb.append(str_or_empty(obj[3]))
                op_time.append(Phrase.merge_as_str(obj[4]))
                op_cond.append(Phrase.merge_as_str(obj[5]))
                op_obj.append(Phrase.merge_as_str(obj[6]))
            else: # type(object) is Phrase
                op.append(str_or_empty(object))
        op = Phrase.merge_as_str(op)
        op_subj = Phrase.merge_as_str(op_subj)
        op_sig = Phrase.merge_as_str(op_sig)
        op_verb = Phrase.merge_as_str(op_verb)
        op_time = Phrase.merge_as_str(op_time)
        op_cond = Phrase.merge_as_str(op_cond)
        op_obj = Phrase.merge_as_str(op_obj)

        return (wp, sp, sw, vp, t, c, op, op_subj, op_sig, op_verb, op_time, op_cond, op_obj)
        
    def as_span(self):
        sp = Phrase.merge(self.subject_phrases).as_span('SUBJECT')
        sw = self.signal_word.as_span('SIGNAL')
        vp = self.verb_phrase.as_span('VERB')
        t = Phrase.merge(self.times).as_span('TIME')
        c = Phrase.merge(self.conditions).as_span('CONDITION')
        
        op, op_subj, op_sig, op_verb, op_time, op_cond, op_obj = [], [], [], [], [], [], []
        for object in self.objects:
            if type(object) is Extracted:
                obj = object.as_tuple()
                op.append(obj[0])
                op_subj.append(Phrase.merge(obj[1]))
                op_sig.append(obj[2])
                op_verb.append(obj[3])
                op_time.append(Phrase.merge(obj[4]))
                op_cond.append(Phrase.merge(obj[5]))
                op_obj.append(Phrase.merge(obj[6]))
            else: # type(object) is Phrase
                op.append(object)
        op = Phrase.merge(op).as_span('OBJECT')
        op_subj = Phrase.merge(op_subj).as_span('OP_SUBJECT')
        op_sig = Phrase.merge(op_sig).as_span('OP_SIGNAL')
        op_verb = Phrase.merge(op_verb).as_span('OP_VERB')
        op_time = Phrase.merge(op_time).as_span('OP_TIME')
        op_cond = Phrase.merge(op_cond).as_span('OP_CONDITION')
        op_obj = Phrase.merge(op_obj).as_span('OP_OBJECT')

        return (sp, sw, vp, t, c, op, op_subj, op_sig, op_verb, op_time, op_cond, op_obj)

    def print(self, display_tree=True):
        sent, subjs, signal, verb, times, conds, op, op_subj, op_signal, op_verb, op_time, op_cond, op_obj = self.as_row()

        print(f'Subjects: {subjs}\nSignal: {signal}\nVerb: {verb}\nTime: {times}\nCondition: {conds}\nObject: {op}\nOP Subject: {op_subj}\nOP Signal: {op_signal}\nOP Verb: {op_verb}\nOP Time: {op_time}\nOP Condition: {op_cond}\nOP Obj: {op_obj}')
        if display_tree:
            displacy.render(self.whole_phrase[0].doc, style="dep", options={"compact": False, "bg": "#09a3d5",
           "color": "white", "font": "Source Sans Pro",
           "collapse_phrases": True})


# helper functions
def by_depth(token):
    return depth_of[token]

def by_index(token):
    return token.i


def phrases_from_roots(roots):
    '''given a list of tokens, return a list of its subtrees as phrases (thus, the tokens are roots of the subtrees)'''
    return [Phrase(root) for root in roots]

def find_root(doc):
    for token in doc:
        if token.dep_.lower() in ['root']:
            return token

def tokens_with_dep(tokens, deps):
    return [t for t in tokens if t.dep_ in deps]

def expand_subtree(token):
    tokens = [child for child in token.subtree]
    tokens.sort(key=by_index) # sort in the order of appearance in the sentence
    return tokens

# depth of each token in the dependency tree (e.g. depth_of[root] = 0)
depth_of = dict()

def assign_depth(token, current_depth=0):
    depth_of[token] = current_depth
    # assign the next depth to all of my children recursively
    for child in token.children:
        assign_depth(child, current_depth+1)

# find the last occurrence of a substring in a string
def last_pos_of(string, substrings):
    last_pos, substr = max([(string.lower().rfind(s), s) for s in substrings], key=lambda pair: pair[0], default=(-1, ''))
    substr = string[last_pos:last_pos+len(substr)]
    return last_pos, substr

# calculate the new shifted index after text replacement at start
def new_index(index, start, old_length, new_length):
    if index <= start:
        return index
    else:
        return index + new_length - old_length

'''
def simplify_sentence(sentence):
    replaced = [] # list of replacements (pos, key)
    # replace all simplifications in reverse order of appearance in the sentence
    while True:
        pos, original = last_pos_of(sentence, simplifications.keys())
        if pos == -1:
            break
        simplified = simplifications[original.lower()]
        sentence = sentence[:pos] + sentence[pos:].replace(original, simplified, 1)
        replaced.append((pos, original))
    return sentence, replaced
'''
'''
def restore_sentence(doc, replaced):
    sentence = doc.text
    doc_json = doc.to_json()

    # cleanse the json to only include the tokens, and spans
    doc_json = {'text': doc_json['text'], 'tokens': doc_json['tokens'], 'spans': doc_json['spans']}

    for pos, original in reversed(replaced):
        simplified = simplifications[original.lower()]
        sentence = sentence[:pos] + sentence[pos:].replace(simplified, original, 1)

        # update all start/end indices of tokens and spans after the replacement
        for token in doc_json['tokens'] + doc_json['spans']['sc']:
            token['start'] = new_index(token['start'], pos, len(simplified), len(original))
            token['end'] = new_index(token['end'], pos, len(simplified), len(original))

    doc_json['text'] = sentence
    return Doc(doc.vocab).from_json(doc_json)
'''

'''
def with_preprocessing(nlp, sentence):
    sentence, replacements = simplify_sentence(sentence)
    doc = nlp(sentence)
    doc = restore_sentence(doc, replacements)
    return doc
'''

def doc_from_annotation(vocab, json):
    json = {
        'text': json['text'],
        'tokens': [{'id': t['id'], 'start': t['start'], 'end': t['end']} for t in json['tokens']],
        'spans': {
            'sc': [{'start': s['start'], 'end': s['end'], 'label': s['label'], 'kb_id': ''} for s in json['spans']]
        }
    }
    doc = Doc(vocab).from_json(json)
    return doc


# add the span labels to the vocabs of nlp
def add_span_label_vocabs(nlp):
    nlp.vocab.strings.add('OP_SUBJECT')
    nlp.vocab.strings.add('OP_SIGNAL')
    nlp.vocab.strings.add('OP_VERB')
    nlp.vocab.strings.add('OP_TIME')
    nlp.vocab.strings.add('OP_CONDITION')
    nlp.vocab.strings.add('OP_OBJECT')

# for displacy span visualization
span_colors = {
    'SUBJECT': '#98DDCA',
    'OBJECT': '#98D6EA',
    'CONDITION': '#F5B5FC',
    'TIME': '#F0F696',
    'SIGNAL': '#C84361',
    'VERB': '#FFAAA7',
    'OP_VERB': '#FFD3B4',
    'OP_SUBJECT': '#D5ECC2',
    'OP_OBJECT': '#BAE5E5',
    'OP_CONDITION': '#F3D1F4',
    'OP_TIME': '#F5FCC1',
    'OP_SIGNAL': '#E78775',
}

# supress CUDA warnings on non-GPU machine
import warnings
warnings.filterwarnings("ignore")