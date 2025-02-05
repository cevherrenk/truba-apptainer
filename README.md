# HDR Reconstruction Projesi

Bu proje, **High Dynamic Range (HDR)** görüntü iyileştirmesi için **Truba** sisteminde **Apptainer** ile özelleştirilmiş bir çalışma ortamı oluşturarak derin öğrenme modelini eğitmeyi amaçlamaktadır. Proje Python 3.8 ve CUDA 11.3 sürümlerini temel alarak geliştirilmiştir. Apptainer sayesinde merkezi Anaconda ortamında yapılamayan özelleştirmeleri gerçekleştirerek, projeye özgü paket sürümleriyle bağımsız bir çalışma ortamı oluşturabilirsiniz.

Truba sistemindeki merkezi Anaconda ortamı, mevcut paketler dışında ek paketlerin kurulumuna izin vermez. Bu durumda, Apptainer size özel ortamlarınızı oluşturma imkanı sağlar. Apptainer, proje için gerekli tüm paketleri içeren bir imaj oluşturmanıza ve bu imaj üzerinden çalışmanıza olanak tanır.

---

## Gereksinimler

- Truba sistemi erişimi
- `apptainer` modülü
- GPU destekli bir ortam

---

## Adımlar

### 1. Apptainer Tanımlama Dosyasını (apptainer.def) ve Betik Dosyalarını Proje Dizininde Oluşturma

Apptainer tanımlama dosyası (`apptainer.def`) ve ilgili betik dosyaları (`apptainer.sh`, `run.sh` gibi), proje dizininiz içinde oluşturulmalıdır. Bu dosyalar, Python sürümü, paket sürümleri ve sistem yapılandırmalarını belirttiğiniz dosyalardır. Dosyalar oluşturulmadan önce proje dizini mevcut değilse aşağıdaki komut ile oluşturulmalıdır:

```bash
mkdir -p /arf/scratch/kullanici_adi/proje_adi
```

Oluşturulan `apptainer.def` dosyası üzerinden `apptainer build` komutu çalıştırıldığında, belirttiğiniz imaj adıyla `.sif` uzantılı bir dosya (örneğin, `python38_torch.sif`) otomatik olarak oluşturulacaktır. `.sif` dosyası elle oluşturulmaz ve komut çalıştırıldıktan sonra ilgili dizinde var olup olmadığı kontrol edilmelidir:

```bash
ls /arf/scratch/kullanici_adi/proje_adi
```

Eğer `.sif` dosyası oluşturulmamışsa, ortam başarıyla oluşmamış demektir ve bu durumda proje çalıştırılamaz. Bu nedenle, komutların doğru bir şekilde çalıştırıldığından emin olunmalıdır.

Bu dosya, Python sürümü ve gerekli paketlerin sürümlerini belirttiğiniz, imajınızı oluşturmak için temel alınan dosyadır.

Aşağıdaki içeriği kullanarak `apptainer.def` adında bir dosya oluşturun:

```dockerfile
Bootstrap: docker
# Kullanılan Python sürümü burada belirtilir. İhtiyacınıza göre 'python:3.8' yerine 'python:3.9' gibi farklı bir sürüm kullanabilirsiniz.
From: python:3.8

%post
    # Sistem güncellemesi ve gerekli araçların kurulumu
# Bu bölümde ortamınız için temel sistem paketleri kurulur. İhtiyaca göre ek paketler eklenebilir veya kaldırılabilir.
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        python3-pip \
        python3-setuptools \
        && rm -rf /var/lib/apt/lists/*

    # Pip'i güncelle
# Python paket yöneticisi olan pip'in güncel sürümünün yüklenmesi sağlanır.
    pip install --upgrade pip

    # PyTorch ve CUDA desteğini yükle
# PyTorch sürümünü ve CUDA uyumlu paketleri buradan belirtebilirsiniz. CUDA sürümünü ihtiyaçlarınıza göre değiştirebilirsiniz.
    pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

    # Ek paketlerin kurulumu
# Projeniz için gerekli Python paketlerini burada listeleyin. Paket sürümlerini belirleyebilir veya ihtiyaçlarınıza göre değiştirebilirsiniz.
    pip install \
        pytorch-lightning==1.4.* \
        piq \
        albumentations \
        pandas \
        tqdm \
        lpips \
        torchmetrics==0.6.0

%environment   
    export PATH=/usr/local/bin:$PATH

%runscript
    # Python ve PyTorch sürümlerini ekrana yazdır
    python -c "
import sys
import torch
import pkg_resources
print('Python Sürümü:', sys.version)
print('Torch Sürümü:', torch.__version__)
print('\nKurulu Paketler ve Sürümleri:')
installed_packages = sorted([f'{d.key}=={d.version}' for d in pkg_resources.working_set])
print('\n'.join(installed_packages))
"
```

### 2. Apptainer Betik Dosyasını (apptainer.sh) Oluşturma ve Ortamın Kalıcı Olarak Oluşturulması

`apptainer.sh` dosyasını aşağıdaki içerikle oluşturun. Bu betik dosyası, Apptainer imajınızı ve ortamınızı kalıcı bir şekilde oluşturur. Ortam bir kez oluşturulduktan sonra tekrar oluşturulmasına gerek yoktur. Betikte süre sınırı, paketlerin kurulum sürelerine bağlı olarak belirlenir. Örnekte maksimum bir saatlik süre sınırı verilmiştir, ancak kurulum daha kısa sürede tamamlanabilir.

Apptainer imajı (`.sif` dosyası) başarıyla oluşturulmuşsa kurulum işlemi başarılı sayılır. Eğer belirtilen süre içinde imaj oluşturulmazsa, süre sınırını artırarak işlemi tekrar deneyebilirsiniz.



apptainer.sh dosyasını aşağıdaki içerikle oluşturun:

```bash
#!/bin/bash
#SBATCH --job-name=hdr   # Kuyruk üzerinde çalişacak iş için isim
#SBATCH -p akya-cuda     # GPU destekli kuyruk adı
#SBATCH -A kullanici_adi # Kullanıcı adı
#SBATCH -J print_gpu     # Gönderilen işin ismi
#SBATCH -o print_gpu.out # Çıktı dosyası
#SBATCH --gres=gpu:1     # Her sunucuda kaç GPU kullanılacak?
#SBATCH -N 1             # Görev kaç node'da çalışacak?
#SBATCH -n 1             # Aynı görevden kaç adet çalıştırılacak?
#SBATCH --cpus-per-task 10  # Her görev kaç CPU kullanacak?
#SBATCH --time=01:00:00     # Süre sınırı
#SBATCH --output=hdr_output.log # Çıktı dosyası
#SBATCH --error=hdr_error.log   # Hata dosyası

# Gerekli modülleri yükle
module load apptainer

# Apptainer cache dizinini ayarla
export APPTAINER_CACHEDIR=/arf/scratch/kullanici_adi/proje_adi/apptainer_cache

# Cache dizinini oluştur
mkdir -p /arf/scratch/kullanici_adi/proje_adi/apptainer_cache

# Çalışma dizinine geç
cd /arf/scratch/kullanici_adi/proje_adi
# Proje dizini burada belirtilmelidir. Eğer dizin mevcut değilse oluşturmanız gereklidir:
# mkdir -p /arf/scratch/kullanici_adi/proje_adi

# Apptainer imajını oluştur
apptainer build imaj_adi.sif apptainer.def
# Bu komut çalıştırıldığında belirttiğiniz 'imaj_adi.sif' dosyası otomatik olarak proje dizininde oluşturulacaktır. Örneğin, 'python38_torch.sif' adı kullanılabilir. Elle oluşturulmaz.
# 'imaj_adi.sif' dosya adı projenize özgü olarak belirlenmelidir. Örneğin, 'python38_torch.sif' gibi bir isim kullanılabilir.

# Apptainer imajını çalıştır ve sürümleri ekrana yazdır
apptainer exec imaj_adi.sif python -c "
import sys
import torch
import pkg_resources

# Python ve PyTorch sürümleri
print('Python Sürümü:', sys.version)
print('Torch Sürümü:', torch.__version__)

# Kurulu olan tüm paketler ve sürümleri
print('\nKurulu Paketler ve Sürümleri:')
installed_packages = sorted([f'{d.key}=={d.version}' for d in pkg_resources.working_set])
print('\n'.join(installed_packages))
"
```

### 3. Ortam Oluşturma ve Projeyi Çalıştırma

Betik dosyalarını ve `apptainer.def` dosyasını oluşturduktan sonra, ortamı oluşturmak için aşağıdaki komutu çalıştırın:

```bash
sbatch apptainer.sh
```

Bu komut, belirttiğiniz süre sınırında ortamı oluşturacak ve `.sif` dosyasını proje dizininde oluşturacaktır. Eğer `apptainer.sh` başarıyla tamamlanır ve `.sif` dosyası belirtilen dizinde oluşmuşsa, ortam oluşturulmuş demektir. Aksi takdirde, hata loglarını kontrol ederek süreyi artırabilir veya betik dosyasında gerekli düzenlemeleri yapabilirsiniz.

İmaj oluşturulduktan sonra aşağıdaki betik dosyasını (örneğin ``run.sh`) hazırlayın.&#x20;

```bash
#!/bin/bash
#SBATCH --job-name=hdr       # Kuyruk üzerinde çalişacak iş için isim
#SBATCH -p akya-cuda          # GPU destekli kuyruk adı
#SBATCH -A kullanici_adi      # Kullanıcı adı
#SBATCH -J print_gpu          # Gönderilen işin ismi
#SBATCH -o print_gpu.out      # Çıktı dosyası
#SBATCH --gres=gpu:4          # Her sunucuda kaç GPU kullanılacak?
#SBATCH -N 1                  # Görev kaç node'da çalışacak?
#SBATCH -n 1                  # Aynı görevden kaç adet çalıştırılacak?
#SBATCH --cpus-per-task 40    # Her görev kaç CPU kullanacak?
#SBATCH --time=03-00:00:00    # Süre sınırı
#SBATCH --output=hdr_output.log # Çıktı dosyası
#SBATCH --error=hdr_error.log   # Hata dosyası

# Gerekli modülleri yükle
module load apptainer

# Çalışma dizinine geç
cd /arf/scratch/kullanici_adi/proje_adi

# Apptainer ortamında projenizi çalıştırın
apptainer exec --nv imaj_adi.sif python train.py
```

```sbatch run.sh``` komutu ile projenizi çalıştırın.

### 4. Projeyi Takip Etme

Çalışma sürecini `sbatch` komutuyla gönderdiğiniz işlemin log dosyaları (örneğin `hdr_output.log` ve `hdr_error.log`) aracılığıyla takip edebilirsiniz.

---

##

