# Final Gold Set V2 Retrieval Report

- Benchmark: `D:\Desktop\legal-rag-malaysia\data\evaluation\final_gold_set_v2.jsonl`
- Vector store: `D:\Desktop\legal-rag-malaysia\data\embeddings\legal-corpus.vectors.jsonl`
- Total queries: 73
- Note: `negative_no_answer_accuracy` is a retrieval-side proxy. A negative query counts as correct only when top-k results avoid surfacing the referenced document as a false answer.

## Overall Results
### lexical
- hit@1: 0.912
- hit@3: 0.985
- wrong-section rate: 0.088
- wrong-document rate: 0.000
- no-answer accuracy for negative queries: 1.000
- failure buckets: {'amendment_failure': 1, 'gazette_failure': 1, 'wrong_unit_in_right_document': 4}

### embedding
- hit@1: 0.441
- hit@3: 0.515
- wrong-section rate: 0.397
- wrong-document rate: 0.162
- no-answer accuracy for negative queries: 1.000
- failure buckets: {'amendment_failure': 3, 'gazette_failure': 1, 'hierarchy_failure': 7, 'wrong_document': 9, 'wrong_unit_in_right_document': 18}

### hybrid
- hit@1: 0.897
- hit@3: 0.985
- wrong-section rate: 0.103
- wrong-document rate: 0.000
- no-answer accuracy for negative queries: 1.000
- failure buckets: {'amendment_failure': 1, 'gazette_failure': 1, 'wrong_unit_in_right_document': 5}

### hybrid_rerank
- hit@1: 1.000
- hit@3: 1.000
- wrong-section rate: 0.000
- wrong-document rate: 0.000
- no-answer accuracy for negative queries: 1.000
- failure buckets: {}

### hybrid_filtered_rerank
- hit@1: 1.000
- hit@3: 1.000
- wrong-section rate: 0.000
- wrong-document rate: 0.000
- no-answer accuracy for negative queries: 1.000
- failure buckets: {}

### graph_supported
- hit@1: 0.912
- hit@3: 1.000
- wrong-section rate: 0.088
- wrong-document rate: 0.000
- no-answer accuracy for negative queries: 1.000
- failure buckets: {'gazette_failure': 1, 'wrong_unit_in_right_document': 5}

### hybrid_plus_graph
- hit@1: 0.912
- hit@3: 1.000
- wrong-section rate: 0.088
- wrong-document rate: 0.000
- no-answer accuracy for negative queries: 1.000
- failure buckets: {'gazette_failure': 1, 'wrong_unit_in_right_document': 5}

### hybrid_plus_graph_with_graph_rerank
- hit@1: 1.000
- hit@3: 1.000
- wrong-section rate: 0.000
- wrong-document rate: 0.000
- no-answer accuracy for negative queries: 1.000
- failure buckets: {}

## Best Mode
- Best hit@1 on this benchmark: `hybrid_rerank`

## Category Breakdown
### lexical
- amendment: count=5, hit@1=0.800, hit@3=0.800, wrong_doc=0.000, wrong_unit=0.200, negative_no_answer=0.000
- bilingual: count=10, hit@1=0.900, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.100, negative_no_answer=0.000
- capability: count=5, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000
- definition: count=8, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000
- direct_lookup: count=10, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000
- gazette_order: count=5, hit@1=0.800, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.200, negative_no_answer=0.000
- heading_lookup: count=10, hit@1=0.700, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.300, negative_no_answer=0.000
- hierarchy: count=8, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000
- negative: count=5, hit@1=0.000, hit@3=0.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=1.000
- obligation: count=3, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000
- rights: count=4, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000

### embedding
- amendment: count=5, hit@1=0.400, hit@3=0.400, wrong_doc=0.000, wrong_unit=0.600, negative_no_answer=0.000
- bilingual: count=10, hit@1=0.200, hit@3=0.200, wrong_doc=0.600, wrong_unit=0.200, negative_no_answer=0.000
- capability: count=5, hit@1=0.600, hit@3=0.600, wrong_doc=0.200, wrong_unit=0.200, negative_no_answer=0.000
- definition: count=8, hit@1=0.875, hit@3=0.875, wrong_doc=0.000, wrong_unit=0.125, negative_no_answer=0.000
- direct_lookup: count=10, hit@1=0.200, hit@3=0.200, wrong_doc=0.100, wrong_unit=0.700, negative_no_answer=0.000
- gazette_order: count=5, hit@1=0.800, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.200, negative_no_answer=0.000
- heading_lookup: count=10, hit@1=0.700, hit@3=0.900, wrong_doc=0.000, wrong_unit=0.300, negative_no_answer=0.000
- hierarchy: count=8, hit@1=0.125, hit@3=0.375, wrong_doc=0.250, wrong_unit=0.625, negative_no_answer=0.000
- negative: count=5, hit@1=0.000, hit@3=0.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=1.000
- obligation: count=3, hit@1=0.333, hit@3=0.333, wrong_doc=0.000, wrong_unit=0.667, negative_no_answer=0.000
- rights: count=4, hit@1=0.250, hit@3=0.250, wrong_doc=0.250, wrong_unit=0.500, negative_no_answer=0.000

### hybrid
- amendment: count=5, hit@1=0.800, hit@3=0.800, wrong_doc=0.000, wrong_unit=0.200, negative_no_answer=0.000
- bilingual: count=10, hit@1=0.900, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.100, negative_no_answer=0.000
- capability: count=5, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000
- definition: count=8, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000
- direct_lookup: count=10, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000
- gazette_order: count=5, hit@1=0.800, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.200, negative_no_answer=0.000
- heading_lookup: count=10, hit@1=0.600, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.400, negative_no_answer=0.000
- hierarchy: count=8, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000
- negative: count=5, hit@1=0.000, hit@3=0.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=1.000
- obligation: count=3, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000
- rights: count=4, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000

### hybrid_rerank
- amendment: count=5, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000
- bilingual: count=10, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000
- capability: count=5, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000
- definition: count=8, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000
- direct_lookup: count=10, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000
- gazette_order: count=5, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000
- heading_lookup: count=10, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000
- hierarchy: count=8, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000
- negative: count=5, hit@1=0.000, hit@3=0.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=1.000
- obligation: count=3, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000
- rights: count=4, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000

### hybrid_filtered_rerank
- amendment: count=5, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000
- bilingual: count=10, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000
- capability: count=5, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000
- definition: count=8, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000
- direct_lookup: count=10, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000
- gazette_order: count=5, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000
- heading_lookup: count=10, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000
- hierarchy: count=8, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000
- negative: count=5, hit@1=0.000, hit@3=0.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=1.000
- obligation: count=3, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000
- rights: count=4, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000

### graph_supported
- amendment: count=5, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000
- bilingual: count=10, hit@1=0.900, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.100, negative_no_answer=0.000
- capability: count=5, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000
- definition: count=8, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000
- direct_lookup: count=10, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000
- gazette_order: count=5, hit@1=0.800, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.200, negative_no_answer=0.000
- heading_lookup: count=10, hit@1=0.600, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.400, negative_no_answer=0.000
- hierarchy: count=8, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000
- negative: count=5, hit@1=0.000, hit@3=0.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=1.000
- obligation: count=3, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000
- rights: count=4, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000

### hybrid_plus_graph
- amendment: count=5, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000
- bilingual: count=10, hit@1=0.900, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.100, negative_no_answer=0.000
- capability: count=5, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000
- definition: count=8, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000
- direct_lookup: count=10, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000
- gazette_order: count=5, hit@1=0.800, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.200, negative_no_answer=0.000
- heading_lookup: count=10, hit@1=0.600, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.400, negative_no_answer=0.000
- hierarchy: count=8, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000
- negative: count=5, hit@1=0.000, hit@3=0.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=1.000
- obligation: count=3, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000
- rights: count=4, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000

### hybrid_plus_graph_with_graph_rerank
- amendment: count=5, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000
- bilingual: count=10, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000
- capability: count=5, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000
- definition: count=8, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000
- direct_lookup: count=10, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000
- gazette_order: count=5, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000
- heading_lookup: count=10, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000
- hierarchy: count=8, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000
- negative: count=5, hit@1=0.000, hit@3=0.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=1.000
- obligation: count=3, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000
- rights: count=4, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000

## Language Breakdown
### lexical
- en: count=59, hit@1=0.907, hit@3=0.981, wrong_doc=0.000, wrong_unit=0.093, negative_no_answer=1.000
- ms: count=14, hit@1=0.929, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.071, negative_no_answer=0.000

### embedding
- en: count=59, hit@1=0.481, hit@3=0.574, wrong_doc=0.074, wrong_unit=0.444, negative_no_answer=1.000
- ms: count=14, hit@1=0.286, hit@3=0.286, wrong_doc=0.500, wrong_unit=0.214, negative_no_answer=0.000

### hybrid
- en: count=59, hit@1=0.889, hit@3=0.981, wrong_doc=0.000, wrong_unit=0.111, negative_no_answer=1.000
- ms: count=14, hit@1=0.929, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.071, negative_no_answer=0.000

### hybrid_rerank
- en: count=59, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=1.000
- ms: count=14, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000

### hybrid_filtered_rerank
- en: count=59, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=1.000
- ms: count=14, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000

### graph_supported
- en: count=59, hit@1=0.907, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.093, negative_no_answer=1.000
- ms: count=14, hit@1=0.929, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.071, negative_no_answer=0.000

### hybrid_plus_graph
- en: count=59, hit@1=0.907, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.093, negative_no_answer=1.000
- ms: count=14, hit@1=0.929, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.071, negative_no_answer=0.000

### hybrid_plus_graph_with_graph_rerank
- en: count=59, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=1.000
- ms: count=14, hit@1=1.000, hit@3=1.000, wrong_doc=0.000, wrong_unit=0.000, negative_no_answer=0.000

## Key Failures
### lexical
- [wrong_unit_in_right_document] Which section of the Employment Act 1955 deals with appeals? (expected=Employment Act 1955/4, actual_top_1=Employment Act 1955 77 (Section 77 Appeal against Director General’s order to High Court))
- [wrong_unit_in_right_document] Which section of the Consumer Protection Act 1999 deals with application? (expected=Consumer Protection Act 1999/2, actual_top_1=Consumer Protection Act 1999 70 (Section 70 Application of other written law))
- [wrong_unit_in_right_document] Which section of the Minimum Wages Order 2024 deals with revocation? (expected=Minimum Wages Order 2024/6, actual_top_1=Minimum Wages Order 2024 3 (Section 3 tonnage, etc. with effect from 1 February 2025))
- [wrong_unit_in_right_document] Seksyen manakah berkaitan cuti tahunan dalam Akta Kerja 1955? (expected=Employment Act 1955/60E, actual_top_1=Employment Act 1955 100 (Section 100 overtime, holidays, annual leave, and sick leave))
- [amendment_failure] When does the Personal Data Protection (Amendment) Act 2024 come into force? (expected=Personal Data Protection (Amendment) Act 2024/1, actual_top_1=Personal Data Protection (Amendment) Act 2024 2 (Section 2 General amendment))
- [gazette_failure] When does the Minimum Wages Order 2024 come into force? (expected=Minimum Wages Order 2024/1, actual_top_1=Minimum Wages Order 2024 3 (Section 3 tonnage, etc. with effect from 1 February 2025))

### embedding
- [wrong_unit_in_right_document] What does Section 10 of the Employment Act 1955 say? (expected=Employment Act 1955/10, actual_top_1=Employment Act 1955 1 (Section 1 Short title and application))
- [wrong_document] Apakah kandungan Perkara 10 dalam Perlembagaan Persekutuan? (expected=Federal Constitution/10, actual_top_1=Employment Act 1955 99A (Section 99A General penalty))
- [wrong_unit_in_right_document] What does Section 10 of the Consumer Protection Act 1999 say? (expected=Consumer Protection Act 1999/10, actual_top_1=Consumer Protection Act 1999 129 (Section 129))
- [wrong_unit_in_right_document] What does Section 10 of the Contracts Act 1950 say? (expected=Contracts Act 1950/10, actual_top_1=Contracts Act 1950 1 (Section 1 Short title))
- [wrong_unit_in_right_document] What does Section 103 of the Income Tax Act 1967 say? (expected=Income Tax Act 1967/103, actual_top_1=Income Tax Act 1967 5 (Section 5 2016, 2017 and 2018))
- [wrong_unit_in_right_document] What does Section 2 of the Minimum Wages Order 2024 say? (expected=Minimum Wages Order 2024/2, actual_top_1=Minimum Wages Order 2024 3 (Section 3 tonnage, etc. with effect from 1 February 2025))
- [wrong_unit_in_right_document] What does Section 10 of the Personal Data Protection (Amendment) Act 2024 say? (expected=Personal Data Protection (Amendment) Act 2024/10, actual_top_1=Personal Data Protection (Amendment) Act 2024 2 (Section 2 General amendment))
- [wrong_unit_in_right_document] What does Section 12 of the Children and Young Persons (Employment) Act 1966 say? (expected=Children and Young Persons (Employment) Act 1966/12, actual_top_1=Children and Young Persons (Employment) Act 1966 1 (Section 1 Short title))
- [wrong_unit_in_right_document] Which section of the Employment Act 1955 deals with appeals? (expected=Employment Act 1955/4, actual_top_1=Employment Act 1955 77 (Section 77 Appeal against Director General’s order to High Court))
- [wrong_unit_in_right_document] Which section of the PDPA sets out the Access Principle? (expected=Personal Data Protection Act 2010/12, actual_top_1=Personal Data Protection Act 2010 81 (Section 81 Procedure))
- [wrong_unit_in_right_document] Which section of the Minimum Wages Order 2024 deals with revocation? (expected=Minimum Wages Order 2024/6, actual_top_1=Minimum Wages Order 2024 3 (Section 3 tonnage, etc. with effect from 1 February 2025))
- [wrong_unit_in_right_document] What is defined under Interpretation in Section 2 of the Income Tax Act 1967? (expected=Income Tax Act 1967/2, actual_top_1=Income Tax Act 1967 79 (Section 79 Interpretation))
- ... and 26 more failed cases in JSON.

### hybrid
- [wrong_unit_in_right_document] Which section of the Employment Act 1955 deals with appeals? (expected=Employment Act 1955/4, actual_top_1=Employment Act 1955 77 (Section 77 Appeal against Director General’s order to High Court))
- [wrong_unit_in_right_document] Which section of the Consumer Protection Act 1999 deals with application? (expected=Consumer Protection Act 1999/2, actual_top_1=Consumer Protection Act 1999 70 (Section 70 Application of other written law))
- [wrong_unit_in_right_document] Which section of the Contracts Act 1950 says acceptance must be absolute? (expected=Contracts Act 1950/7, actual_top_1=Contracts Act 1950 164 (Section 164 Agent’s duty in conducting principal’s business))
- [wrong_unit_in_right_document] Which section of the Minimum Wages Order 2024 deals with revocation? (expected=Minimum Wages Order 2024/6, actual_top_1=Minimum Wages Order 2024 3 (Section 3 tonnage, etc. with effect from 1 February 2025))
- [wrong_unit_in_right_document] Seksyen manakah berkaitan cuti tahunan dalam Akta Kerja 1955? (expected=Employment Act 1955/60E, actual_top_1=Employment Act 1955 100 (Section 100 overtime, holidays, annual leave, and sick leave))
- [amendment_failure] When does the Personal Data Protection (Amendment) Act 2024 come into force? (expected=Personal Data Protection (Amendment) Act 2024/1, actual_top_1=Personal Data Protection (Amendment) Act 2024 2 (Section 2 General amendment))
- [gazette_failure] When does the Minimum Wages Order 2024 come into force? (expected=Minimum Wages Order 2024/1, actual_top_1=Minimum Wages Order 2024 3 (Section 3 tonnage, etc. with effect from 1 February 2025))

### hybrid_rerank
- No failed cases.

### hybrid_filtered_rerank
- No failed cases.

### graph_supported
- [wrong_unit_in_right_document] Which section of the Employment Act 1955 deals with appeals? (expected=Employment Act 1955/4, actual_top_1=Employment Act 1955 77 (Section 77 Appeal against Director General’s order to High Court))
- [wrong_unit_in_right_document] Which section of the Consumer Protection Act 1999 deals with application? (expected=Consumer Protection Act 1999/2, actual_top_1=Consumer Protection Act 1999 70 (Section 70 Application of other written law))
- [wrong_unit_in_right_document] Which section of the Contracts Act 1950 says acceptance must be absolute? (expected=Contracts Act 1950/7, actual_top_1=Contracts Act 1950 164 (Section 164 Agent’s duty in conducting principal’s business))
- [wrong_unit_in_right_document] Which section of the Minimum Wages Order 2024 deals with revocation? (expected=Minimum Wages Order 2024/6, actual_top_1=Minimum Wages Order 2024 3 (Section 3 tonnage, etc. with effect from 1 February 2025))
- [wrong_unit_in_right_document] Seksyen manakah berkaitan cuti tahunan dalam Akta Kerja 1955? (expected=Employment Act 1955/60E, actual_top_1=Employment Act 1955 100 (Section 100 overtime, holidays, annual leave, and sick leave))
- [gazette_failure] When does the Minimum Wages Order 2024 come into force? (expected=Minimum Wages Order 2024/1, actual_top_1=Minimum Wages Order 2024 3 (Section 3 tonnage, etc. with effect from 1 February 2025))

### hybrid_plus_graph
- [wrong_unit_in_right_document] Which section of the Employment Act 1955 deals with appeals? (expected=Employment Act 1955/4, actual_top_1=Employment Act 1955 77 (Section 77 Appeal against Director General’s order to High Court))
- [wrong_unit_in_right_document] Which section of the Consumer Protection Act 1999 deals with application? (expected=Consumer Protection Act 1999/2, actual_top_1=Consumer Protection Act 1999 70 (Section 70 Application of other written law))
- [wrong_unit_in_right_document] Which section of the Contracts Act 1950 says acceptance must be absolute? (expected=Contracts Act 1950/7, actual_top_1=Contracts Act 1950 164 (Section 164 Agent’s duty in conducting principal’s business))
- [wrong_unit_in_right_document] Which section of the Minimum Wages Order 2024 deals with revocation? (expected=Minimum Wages Order 2024/6, actual_top_1=Minimum Wages Order 2024 3 (Section 3 tonnage, etc. with effect from 1 February 2025))
- [wrong_unit_in_right_document] Seksyen manakah berkaitan cuti tahunan dalam Akta Kerja 1955? (expected=Employment Act 1955/60E, actual_top_1=Employment Act 1955 100 (Section 100 overtime, holidays, annual leave, and sick leave))
- [gazette_failure] When does the Minimum Wages Order 2024 come into force? (expected=Minimum Wages Order 2024/1, actual_top_1=Minimum Wages Order 2024 3 (Section 3 tonnage, etc. with effect from 1 February 2025))

### hybrid_plus_graph_with_graph_rerank
- No failed cases.

