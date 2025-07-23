# SPDX-FileCopyrightText: 2024 PeARS Project, <community@pearsproject.org>, 
#
# SPDX-License-Identifier: AGPL-3.0-only

import math
import logging
from time import time
from os import getenv
from os.path import dirname, join, realpath
from itertools import islice
from urllib.parse import urlparse
from glob import glob
from collections import Counter
import joblib
from joblib import Parallel, delayed
from scipy.spatial import distance
from scipy.sparse import load_npz, csr_matrix, vstack
import numpy as np
from flask import url_for
from flask_login import current_user
from app import app, db, models
from app.api.models import Urls
from app.search.overlap_calculation import (snippet_overlap,
        score_url_overlap, posix, posix_no_seq)
from app.utils import parse_query, timer
from app.indexer.mk_page_vector import compute_query_vectors, tokenize_text
from app.indexer.posix import load_posix

dir_path = dirname(dirname(realpath(__file__)))
pod_dir = getenv("PODS_DIR", join(dir_path, 'pods'))

def mk_podsum_matrix(lang):
    """ Make the podsum matrix, i.e. a matrix
    with each row corresponding to the sum of 
    all documents in a given pod."""
    podnames = []
    podsum = []
    npzs = glob(join(pod_dir,'*',lang,'*.u.*npz'))
    for npz in npzs:
        podname = npz.split('/')[-1].replace('.npz','')
        s = np.sum(load_npz(npz).toarray(), axis=0)
        #print(podname, np.sum(s), s)
        if np.sum(s) > 0:
            podsum.append(s)
            podnames.append(podname)
    return podnames, podsum





@timer
def mk_vec_matrix(lang):
    """ Make a vector matrix by stacking all
    pod matrices."""
    c = 0
    podnames = []
    bins = [c]
    m = []
    urls = []

    npzs = glob(join(pod_dir,'*',lang,'*.u.*npz'))
    for npz in npzs:
        podnames.append(npz.split('/')[-1].replace('.npz',''))

    # deal with languages for which no pods exist (yet)
    if not podnames:
        return None, [], [], []

    with app.app_context():
        for i in range(len(podnames)):
            us = db.session.query(Urls).filter_by(pod=podnames[i]).all()
            if len(us) == 0:
                continue
            upaths = [u.url for u in us if u.vector is not None]
            idvs = [u.vector for u in us if u.vector is not None]
            urls.extend(upaths)
            npz = load_npz(npzs[i]).toarray()
            try:
                npz = npz[idvs,:]
                m.append(csr_matrix(npz))
            except IndexError as e:
                raise RuntimeError(f"pod {npzs[i]} seems to be corrupt - {e}")
            c+=npz.shape[0]
            bins.append(c)
        try:
            m = vstack(m)
        except ValueError as e:
            raise RuntimeError(f"error stacking these pods: {npzs} - {e}")
        m = csr_matrix(m)
    return m, bins, podnames, urls


def load_vec_matrix(lang):
    if 'm' in models[lang]:
        m = models[lang]['m']
        bins = models[lang]['mbins']
        podnames = models[lang]['podnames']
        urls = models[lang]['urls']
    else:
        m, bins, podnames, urls = mk_vec_matrix(lang)
    if m is not None:
        m = m.todense()
    return m, bins, podnames, urls



@timer
def compute_scores(query, query_vectors, lang):
    # extended snippets or not? 
    if app.config["EXTENDED_SNIPPETS_WHEN_LOGGED_IN"] and current_user.is_authenticated:
        use_extended_snippets = True
        snippet_length = app.config['EXTENDED_SNIPPET_LENGTH']
    else:
        use_extended_snippets = False
        snippet_length = app.config['SNIPPET_LENGTH']
    m, bins, podnames, urls = load_vec_matrix(lang)
    if m is None:
        return {}
    query_vector = np.sum(query_vectors, axis=0)
    
    # Only compute cosines over the dimensions of interest
    a = np.where(query_vector!=0)[1]
    cos = 1 - distance.cdist(query_vector[:,a], m[:,a], 'cosine')[0]
    cos[np.isnan(cos)] = 0

    # Document ids with non-zero values (match at least one subword)
    idx = np.where(cos!=0)[0]

    # Sort document ids with non-zero values and take top 50
    idx = np.argsort(cos)[-len(idx):][::-1][:50]

    # Get urls
    document_scores = {}
    best_urls = [urls[i] for i in idx]
    best_cos = [cos[i] for i in idx]
    us = Urls.query.filter(Urls.url.in_(best_urls)).all()

    snippet_scores = {}
    for u in us:
        if use_extended_snippets:
            snippet = u.extended_snippet
        else:
            snippet = u.snippet

        if snippet is None:
            snippet = ''
            snippet_score = 0.0
        else:
            snippet = ' '.join(snippet.split()[:snippet_length])
            snippet_score = snippet_overlap(query, u.title+' '+snippet)
        loc = urlparse(u.url).netloc.split('.')[0]

        #Big boost in case the query word is the url
        if query == loc:
            snippet_score+=0.5
        #Little boost in case the query words are in the url
        for w in query.split():
            if w in u.url:
                snippet_score+=0.1
        snippet_scores[u.url] = snippet_score

    for i, u in enumerate(best_urls):
        #print(f"url: {u}, snippet_score: {snippet_scores[u]}, cos: {best_cos[i]}")
        document_scores[u] = best_cos[i] + snippet_scores[u]

    return document_scores


def return_best_urls(doc_scores):
    best_urls = []
    scores = []
    #netlocs_used = []  # Don't return 100 pages from the same site
    c = 0
    for w in sorted(doc_scores, key=doc_scores.get, reverse=True):
        #loc = urlparse(w).netloc
        if c < 50:
            if doc_scores[w] >= 0.5:
                #if netlocs_used.count(loc) < 10:
                #print("DOC SCORE",w,doc_scores[w])
                best_urls.append(w)
                scores.append(doc_scores[w])
                #netlocs_used.append(loc)
                c += 1
            else:
                break
        else:
            break
    return best_urls, scores


def output(best_urls, scores):
    # extended snippets or not? 
    if app.config["EXTENDED_SNIPPETS_WHEN_LOGGED_IN"] and current_user.is_authenticated:
        use_extended_snippets = True
        snippet_length = app.config['EXTENDED_SNIPPET_LENGTH']
    else:
        use_extended_snippets = False
        snippet_length = app.config['SNIPPET_LENGTH']
    results = {}
    urls = Urls.query.filter(Urls.url.in_(best_urls)).all()
    urls = [next(u for u in urls if u.url == best_url) for best_url in best_urls]
    for i, u in enumerate(urls):
        url = u.url
        results[url] = u.as_dict()
        results[url]['score'] = scores[i]
        if not url.startswith('pearslocal'):
            if use_extended_snippets:
                results[url]['snippet'] = ' '.join(results[url]['extended_snippet'].split()[:snippet_length])
            else:
                results[url]['snippet'] = ' '.join(results[url]['snippet'].split()[:snippet_length])
    return results



def run_search(query, lang, extended=True):
    """Run search on query input by user

    Parameter: query, a query string.
    Returns: a list of documents. Each document is a dictionary. 
    """
    document_scores = {}
    extended_document_scores = {}

    # Run tokenization and vectorization on query. We also get an extended query and its vector.
    q_tokenized, extended_q_tokenized, q_vectors, extended_q_vectors = compute_query_vectors(query, lang, expansion_length=10)

    document_scores = compute_scores(query, q_vectors, lang)

    if extended:
        extended_document_scores = compute_scores(query, extended_q_vectors, lang)

    # Merge extended results
    merged_scores = document_scores.copy()
    for k,_ in extended_document_scores.items():
        if k in document_scores:
            merged_scores[k] = document_scores[k]+ 0.5*extended_document_scores[k]
        else:
            merged_scores[k] = 0.5*extended_document_scores[k]

    # if posix is enabled, search for best pods and perform posix search on those pods
    if app.config["ENABLE_POSIX"]:
        posix_best_urls = {}
        best_pods = score_pods(q_vectors, extended_q_vectors, lang, max_pods=10)
        for pod in best_pods:
            theme, lang_and_user = pod.split(".l.")
            _, user = pod.split(".u.")
            posix = load_posix(user, lang, theme)
            posix_best_doc_ids = intersect_best_posix_lists(q_tokenized, posix, lang)
            for doc_id, posix_score in posix_best_doc_ids.items():
                doc_url = db.session.query(Urls).filter_by(pod=pod, vector=doc_id).first()
                posix_best_urls[doc_url.url] = posix_score

        # Merge posix & vector results
        combined_urls = set(merged_scores.keys()).union(set(posix_best_urls.keys()))
        combined_vector_and_posix_scores = {}
        for url in combined_urls:
            mean_score = (posix_best_urls.get(url, 0.0) + merged_scores.get(url, 0.0)) / 2
            combined_vector_and_posix_scores[url] = mean_score

        best_urls, scores = return_best_urls(combined_vector_and_posix_scores)

    else:
        best_urls, scores = return_best_urls(merged_scores)

    results = output(best_urls, scores)
    return results, scores



def intersect_best_posix_lists(query_tokenized, posindex, lang):
    tmp_best_docs = []
    posix_scores = {}
    # Loop throught the token list corresponding to each word
    for word_tokens in query_tokenized:
        scores = posix(' '.join(word_tokens), posindex, lang)
        logging.debug(f"POSIX SCORES: {scores}")
        tmp_best_docs.append(list(scores.keys()))
        for k,v in scores.items():
            if k in posix_scores:
                posix_scores[k].append(v)
            else:
                posix_scores[k] = [v]
    q_best_docs = set.intersection(*map(set,tmp_best_docs))
    if len(q_best_docs) == 0:
        q_best_docs = set.union(*map(set,tmp_best_docs))
    best_docs = {}
    for d in q_best_docs:
        docscore = np.mean(posix_scores[d])
        best_docs[d] = docscore
    logging.info(f"BEST DOCS FROM POS INDEX: {best_docs}")
    return best_docs

def make_posix_extended_snippet(query, url, idv, pod, context=4, max_length=100):
    snippet = []
    
    theme, lang_and_user = pod.split(".l.")
    lang, user = lang_and_user.split(".u.")
    
    idv = int(idv)

    query_tokenized = []
    for w in query.split():
        query_tokenized.extend(tokenize_text(w, lang, stringify=False))

    vocab = models[lang]['vocab']
    inverted_vocab = models[lang]['inverted_vocab']
    query_vocab_ids = [vocab.get(wp) for wp in query_tokenized]
    if any([i is None for i in query_vocab_ids]):
        query_vocab_ids = [i for i in query_vocab_ids if i is not None]
    
    posix = load_posix(user, lang, theme)
    
    spans = []

    # reconstruct the original document so we can fill in the snippets beyond
    pos_to_wp = {}
    for vid in range(len(vocab)):
        doc_positions = [int(pos_str) for pos_str in posix[vid].get(idv, "").split("|") if pos_str != ""]
        for pos in doc_positions:
            pos_to_wp[int(pos)] = inverted_vocab[vid]
    doc_length = max(pos_to_wp.keys()) 

    for wp, vocab_id in zip(query_tokenized, query_vocab_ids):
        
        # all of the positions of this wp in the doc
        positions = [int(pos) for pos in posix[vocab_id].get(idv, "").split("|") if pos]

        # check if the position is adjacent to any of the existing spans
        for span_idx in range(len(spans)):

            span_start, span_end = spans[span_idx]

            for pos in positions:
                # attach to the span if it follows directly after the previous token
                if pos - 1 == span_end:
                    spans[span_idx] = (span_start, pos)
                    break
                    
        # start-of-word token not yet included in existing spans: start new spans
        if wp.startswith("▁"):
            for pos in positions:
                in_existing_span = False
                for (span_start, span_end) in spans:
                    if pos in range(span_start, span_end+1):
                        in_existing_span = True
                        break
                if not in_existing_span:
                    spans.append((pos, pos))

    # add a window of context to the spans, merge spans if needed
    extended_spans = []
    for span_idx in range(len(spans)):        
        span_start, span_end = spans[span_idx]
        
        overlaps_with_existing_span = False
        for xspan_id in range(len(extended_spans)):
            xspan_start, xspan_end = extended_spans[xspan_id]
            if span_start in range(xspan_start, xspan_end+1) or span_end in range(xspan_start, xspan_end+1):
                overlaps_with_existing_span = True
                extended_spans[xspan_id] = min(xspan_start, span_start), max(xspan_end, span_end)
                break

        if not overlaps_with_existing_span:
            xspan_start = max(0, span_start - context)
            xspan_end = min(doc_length, span_end + context)
            extended_spans.append((xspan_start, xspan_end))  

    for span_start, span_end in extended_spans:
        span_txt = []
        for i in range(span_start, span_end+1):
            span_txt.append(pos_to_wp.get(i, " "))
        snippet.append("".join(span_txt).replace("▁", " ").lstrip())
    snippet_str = " ... ".join(snippet)
    snippet_capped = " ".join(snippet_str.split()[:max_length])
    if not snippet_capped.endswith("..."):
        return snippet_capped + "..."
    return snippet_capped


@timer
def score_pods(query_vectors, extended_q_vectors, lang, max_pods=3):
    """Score pods for a query.

    Parameters:
    query_vector: the numpy array for the query (dim = size of vocab)
    extended_q_vectors: a list of numpy arrays for the extended query
    lang: the language of the query

    Returns: a list of the best <max_pods: int> pods.
    """
    print(">> SEARCH: SCORE PAGES: SCORE PODS")

    pod_scores = {}

    m, bins, podnames, urls = load_vec_matrix(lang)
    if m is None:
        return {}

    tmp_best_pods = []
    tmp_best_scores = []
    # For each word in the query, compute best pods
    for query_vector in query_vectors:
        # Only compute cosines over the dimensions of interest
        a = np.where(query_vector!=0)[1]
        cos = 1 - distance.cdist(query_vector[:,a], m[:,a], 'cosine')[0]
        cos[np.isnan(cos)] = 0

        # Document ids with non-zero values (match at least one subword)
        idx = np.where(cos!=0)[0]

        # Sort document ids with non-zero values
        idx = np.argsort(cos)[-len(idx):][::-1]

        # Bin document ids into pods, and record how many documents are matched in each bin
        d = np.digitize(idx, bins)
        d = dict(Counter(d).most_common())
        best_bins = list(d.keys())
        best_bins = [b-1 for b in best_bins] #digitize starts at 1, not 0
        print(best_bins)
        best_scores = list(d.values())
        max_score = max(best_scores)
        best_scores = np.array(best_scores) / max_score

        #pods = [podnames[b] for b in best_bins]
        tmp_best_pods.append(best_bins)
        tmp_best_scores.append(best_scores)

    best_pods = {}
    best_pods_urls = {}
    maximums = np.ones((1,len(query_vectors)))
    scores = np.zeros((1,len(query_vectors)))
    for p in range(len(podnames)):
        podname = podnames[p]
        pod_urls = urls[p]
        for i, arr in enumerate(tmp_best_pods):
            score = tmp_best_scores[i][tmp_best_pods[i].index(p)] if p in tmp_best_pods[i] else 0
            scores[0][i] = score
        podscore = 1 - distance.cdist(maximums,scores, 'euclidean')[0][0]
        #if podscore != 0:
        #    print(f"POD {podnames[p]} {scores} {podscore}")
        best_pods[podname] = podscore
    best_pods = dict(sorted(best_pods.items(), key=lambda item: item[1], reverse=True))
    best_pods = dict(islice(best_pods.items(), max_pods))
    best_pods = list(best_pods.keys())
    return best_pods

