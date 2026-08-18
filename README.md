# Bridging the Data Gap in Incremental Object Counting via Text-Semantic Adaptation


# Abstract

Incremental object counting seeks to estimate object quantities in dynamically evolving scenes while continuously adapting to novel categories without forgetting previously acquired knowledge. Nevertheless, the high cost of annotating dense scenes in real-world applications often results in insufficient training data during each incremental stage, which further exacerbates the learning challenge. To overcome this limitation, we propose a text-semantic adaptation framework for incremental object counting that leverages complementary semantic knowledge beyond visual supervision. Specifically, we introduce a structure-anchored modulation adapter to efficiently adapt pre-trained models by dynamically generating task-specific modulation weights for coordinating category-specific incremental adapters. Moreover, we propose an incremental cross-semantic weaver that builds heterogeneous relational graphs from visual and textual modalities and enhances cross-modal representation learning through graph interactions, thereby improving feature discrimination under limited incremental supervision. Extensive experiments demonstrate that the proposed method attains state-of-the-art performance.

