# Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (2nd Edition)

## Gambaran Umum Repository

Repository ini berisi implementasi dan contoh-contoh praktis dari buku "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" Edisi Kedua oleh Aurélien Géron. Buku ini menyediakan pengenalan komprehensif tentang machine learning dan deep learning menggunakan framework Python yang paling populer.

## Struktur Buku dan Ringkasan Bab

### **Bagian I: Dasar-Dasar Machine Learning**

#### Bab 1: The Machine Learning Landscape
- **Gambaran Umum**: Pengenalan konsep machine learning, jenis-jenis sistem ML, dan lanskap secara keseluruhan
- **Topik Utama**: Supervised vs unsupervised learning, batch vs online learning, instance-based vs model-based learning
- **Fokus Praktis**: Memahami kapan dan bagaimana menerapkan pendekatan ML yang berbeda
- **Tools**: Setup dasar Python dan pengenalan workflow ML

#### Bab 2: End-to-End Machine Learning Project
- **Gambaran Umum**: Panduan lengkap proyek ML dunia nyata dari awal hingga akhir
- **Topik Utama**: Eksplorasi data, visualisasi, preprocessing, pemilihan model, dan evaluasi
- **Fokus Praktis**: Prediksi harga rumah California menggunakan data real estate
- **Tools**: Pandas, Matplotlib, dasar-dasar Scikit-Learn

#### Bab 3: Classification
- **Gambaran Umum**: Pengenalan algoritma dan teknik klasifikasi
- **Topik Utama**: Klasifikasi biner dan multiclass, metrik performa, confusion matrices
- **Fokus Praktis**: Klasifikasi digit MNIST, trade-off precision/recall
- **Tools**: Classifier Scikit-Learn (SGD, Random Forest, SVM)

#### Bab 4: Training Models
- **Gambaran Umum**: Memahami bagaimana algoritma ML belajar dan dasar matematisnya
- **Topik Utama**: Regresi linear, gradient descent, regresi logistik, regularisasi
- **Fokus Praktis**: Intuisi matematis di balik algoritma pembelajaran
- **Tools**: Implementasi NumPy dan perbandingan Scikit-Learn

#### Bab 5: Support Vector Machines
- **Gambaran Umum**: Pembahasan mendalam algoritma SVM untuk klasifikasi dan regresi
- **Topik Utama**: Linear SVM, nonlinear SVM, kernel trick, SVM regression
- **Fokus Praktis**: Memahami decision boundaries dan fungsi kernel
- **Tools**: Implementasi SVM Scikit-Learn

#### Bab 6: Decision Trees
- **Gambaran Umum**: Algoritma berbasis pohon untuk klasifikasi dan regresi
- **Topik Utama**: Konstruksi decision tree, pruning, feature importance
- **Fokus Praktis**: Klasifikasi dataset Iris, visualisasi pohon
- **Tools**: DecisionTreeClassifier dan DecisionTreeRegressor Scikit-Learn

#### Bab 7: Ensemble Learning and Random Forests
- **Gambaran Umum**: Menggabungkan beberapa model untuk performa yang lebih baik
- **Topik Utama**: Voting classifiers, bagging, pasting, random forests, boosting
- **Fokus Praktis**: Implementasi AdaBoost, Gradient Boosting, dan Random Forest
- **Tools**: Metode ensemble Scikit-Learn

#### Bab 8: Dimensionality Reduction
- **Gambaran Umum**: Teknik mengurangi jumlah fitur sambil mempertahankan informasi
- **Topik Utama**: PCA, kernel PCA, LLE, t-SNE, feature selection
- **Fokus Praktis**: Visualisasi data berdimensi tinggi, curse of dimensionality
- **Tools**: Algoritma dimensionality reduction Scikit-Learn

#### Bab 9: Unsupervised Learning Techniques
- **Gambaran Umum**: Pembelajaran pola dari data tanpa label
- **Topik Utama**: K-means clustering, hierarchical clustering, DBSCAN, Gaussian mixtures
- **Fokus Praktis**: Segmentasi pelanggan, deteksi anomali
- **Tools**: Algoritma clustering Scikit-Learn

### **Bagian II: Neural Networks dan Deep Learning**

#### Bab 10: Introduction to Artificial Neural Networks with Keras
- **Gambaran Umum**: Dasar-dasar neural networks dan deep learning
- **Topik Utama**: Perceptron, multilayer perceptrons, backpropagation
- **Fokus Praktis**: Membangun neural network pertama untuk klasifikasi gambar
- **Tools**: TensorFlow 2.0 dan Keras

#### Bab 11: Training Deep Neural Networks
- **Gambaran Umum**: Teknik untuk melatih deep networks secara efektif
- **Topik Utama**: Vanishing gradients, activation functions, weight initialization, batch normalization
- **Fokus Praktis**: Strategi optimisasi deep network
- **Tools**: Teknik Keras lanjutan, TensorBoard

#### Bab 12: Custom Models and Training with TensorFlow
- **Gambaran Umum**: Membangun arsitektur neural network kustom
- **Topik Utama**: API tingkat rendah TensorFlow, custom layers, loss functions, metrics
- **Fokus Praktis**: Membuat komponen ML yang fleksibel dan dapat digunakan kembali
- **Tools**: Operasi tingkat rendah TensorFlow 2.0

#### Bab 13: Loading and Preprocessing Data with TensorFlow
- **Gambaran Umum**: Pipeline data yang efisien untuk machine learning
- **Topik Utama**: tf.data API, preprocessing data, feature engineering
- **Fokus Praktis**: Menangani dataset besar, data augmentation
- **Tools**: TensorFlow Data API, preprocessing layers

#### Bab 14: Deep Computer Vision Using Convolutional Neural Networks
- **Gambaran Umum**: Arsitektur CNN untuk pengenalan gambar dan computer vision
- **Topik Utama**: Convolution, pooling, arsitektur CNN (LeNet, AlexNet, ResNet)
- **Fokus Praktis**: Klasifikasi gambar, dasar-dasar object detection
- **Tools**: Layer CNN Keras, transfer learning

#### Bab 15: Processing Sequences Using RNNs and CNNs
- **Gambaran Umum**: Neural networks untuk pemrosesan data sekuensial
- **Topik Utama**: Vanilla RNN, LSTM, GRU, sequence-to-sequence models
- **Fokus Praktis**: Forecasting time series, analisis sentimen
- **Tools**: Layer RNN Keras, preprocessing sequence

#### Bab 16: Natural Language Processing with RNNs and Attention
- **Gambaran Umum**: Teknik NLP lanjutan menggunakan neural networks
- **Topik Utama**: Word embeddings, encoder-decoder models, attention mechanisms
- **Fokus Praktis**: Machine translation, text generation
- **Tools**: Layer embedding Keras, implementasi attention

#### Bab 17: Representation Learning and Generative Learning Using Autoencoders
- **Gambaran Umum**: Unsupervised learning dengan neural networks
- **Topik Utama**: Autoencoders, variational autoencoders, denoising autoencoders
- **Fokus Praktis**: Dimensionality reduction, generative modeling
- **Tools**: Implementasi autoencoder Keras

#### Bab 18: Reinforcement Learning
- **Gambaran Umum**: Pembelajaran melalui interaksi dengan lingkungan
- **Topik Utama**: Q-learning, policy gradients, deep reinforcement learning
- **Fokus Praktis**: Agen bermain game, aplikasi robotika
- **Tools**: TensorFlow Agents, OpenAI Gym

#### Bab 19: Training and Deploying TensorFlow Models at Scale
- **Gambaran Umum**: Deployment produksi dan scaling model ML
- **Topik Utama**: TensorFlow Serving, optimisasi model, distributed training
- **Fokus Praktis**: Strategi deployment dunia nyata
- **Tools**: TensorFlow Extended (TFX), cloud deployment

## Struktur Repository

```
├── notebooks/
│   ├── 01_the_machine_learning_landscape/
│   ├── 02_end_to_end_machine_learning_project/
│   ├── 03_classification/
│   ├── 04_training_models/
│   ├── 05_support_vector_machines/
│   ├── 06_decision_trees/
│   ├── 07_ensemble_learning_and_random_forests/
│   ├── 08_dimensionality_reduction/
│   ├── 09_unsupervised_learning/
│   ├── 10_neural_nets_with_keras/
│   ├── 11_training_deep_neural_networks/
│   ├── 12_custom_models_and_training/
│   ├── 13_loading_and_preprocessing_data/
│   ├── 14_deep_computer_vision/
│   ├── 15_processing_sequences/
│   ├── 16_nlp_with_rnns_and_attention/
│   ├── 17_autoencoders/
│   ├── 18_reinforcement_learning/
│   └── 19_training_and_deploying_at_scale/
└── README.md
```

## Tujuan Pembelajaran Utama

### Traditional Machine Learning (Bab 1-9)
- Menguasai konsep dan algoritma ML fundamental
- Memahami kapan menerapkan algoritma yang berbeda
- Mempelajari preprocessing data dan feature engineering yang tepat
- Mengimplementasikan proyek ML end-to-end dengan Scikit-Learn
- Menangani masalah supervised dan unsupervised learning

### Deep Learning (Bab 10-19)
- Membangun dan melatih neural networks dengan TensorFlow/Keras
- Mengimplementasikan convolutional neural networks untuk computer vision
- Bekerja dengan recurrent neural networks untuk data sekuensial
- Menerapkan attention mechanisms dan teknik NLP lanjutan
- Membuat generative models dengan autoencoders
- Mengeksplorasi algoritma reinforcement learning
- Deploy model di lingkungan produksi

## Acknowledgments

- **Aurélien Géron**: Penulis buku asli
- **O'Reilly Media**: Penerbit
- **Open Source Community**: Untuk tools dan libraries luar biasa yang digunakan
