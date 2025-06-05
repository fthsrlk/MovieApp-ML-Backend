document.addEventListener('DOMContentLoaded', function() {
  // Watchlist işlemleri
  setupWatchlistButtons();
  
  // Derecelendirme işlemleri
  setupRatingSystem();
  
  // Tab yönetimi
  setupTabs('.tabs', '.tab', '.tab-content');
  
  // Tüm Verileri Temizle butonu
  setupClearAllDataButton();
  // createHeroSparkles(); // Artık SPA içinde yönetilecek
  // setupFullpageScroll(); // Artık SPA içinde yönetilecek
  // Ana sayfa (page-index) ilk yüklendiğinde tam sayfa scroll ve animasyonları başlat
  if (document.body.classList.contains('page-index')) {
    createHeroSparkles();
    setupFullpageScroll();
    setupDotNavigation();
  }
  initSpaNavigation(); // Yeni SPA navigasyonunu başlat
  setupHamburgerMenu();
  // if (document.body.contains('page-index')) { // Artık SPA içinde yönetilecek
  //   setupDotNavigation(); 
  // }

  // Dinamik bölümler görünürlüğe girdiğinde yükle
  setupDynamicSectionLoader();

  if (window.location.pathname.includes('/recommendations')) {
    console.log("Recommendations page direct load: Initializing tabs.");
    setupTabs('.ml-tabs', '.recommendations-tab', '.recommendations-content');
    setupTabs('.tmdb-tabs', '.recommendations-tab', '.recommendations-content');
  }
});

// Tüm Verileri Temizle butonu
function setupClearAllDataButton() {
  const clearAllDataBtn = document.getElementById('clearAllDataBtn');
  if (clearAllDataBtn) {
    clearAllDataBtn.addEventListener('click', function() {
      if (confirm('Tüm verileriniz temizlenecek. Bu işlem geri alınamaz. Devam etmek istiyor musunuz?')) {
        fetch('/api/clear_all_data', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({})
        })
        .then(response => response.json())
        .then(data => {
          if (data.success) {
            showNotification('Tüm veriler başarıyla temizlendi!');
            // Kullanıcıyı ana sayfaya yönlendir
            setTimeout(() => {
              window.location.href = '/';
            }, 1500);
          } else {
            showNotification('Veriler temizlenirken bir hata oluştu: ' + data.error, 'error');
          }
        })
        .catch(error => {
          console.error('Hata:', error);
          showNotification('Bir hata oluştu, lütfen tekrar deneyin.', 'error');
        });
      }
    });
  }
}

// İzleme listesi butonları (Detay sayfası ve Kartlar için)
function setupWatchlistButtons() {
  // Detay Sayfasındaki Butonlar (ID ile seçilir)
  const detailAddToWatchlistBtn = document.getElementById('add-to-watchlist');
  const detailRemoveFromWatchlistBtn = document.getElementById('remove-from-watchlist');

  if (detailAddToWatchlistBtn) {
    detailAddToWatchlistBtn.addEventListener('click', function() {
      handleWatchlistAction(this, 'add');
    });
  }

  if (detailRemoveFromWatchlistBtn) {
    detailRemoveFromWatchlistBtn.addEventListener('click', function() {
      handleWatchlistAction(this, 'remove');
    });
  }

  // Kartlardaki Butonlar (Sınıf ile seçilir)
  const cardAddToWatchlistBtns = document.querySelectorAll('.add-to-watchlist-card');
  cardAddToWatchlistBtns.forEach(btn => {
    btn.addEventListener('click', function() {
      handleWatchlistAction(this, 'add', true /* isCardButton */);
    });
  });

  const cardRemoveFromWatchlistBtns = document.querySelectorAll('.remove-from-watchlist-card');
  cardRemoveFromWatchlistBtns.forEach(btn => {
    btn.addEventListener('click', function() {
      handleWatchlistAction(this, 'remove', true /* isCardButton */);
    });
  });
}

// Genel izleme listesi işlem fonksiyonu
function handleWatchlistAction(buttonElement, action, isCardButton = false) {
  const mediaType = buttonElement.dataset.mediaType;
  const itemId = parseInt(buttonElement.dataset.itemId, 10);
  const apiUrl = (action === 'add') ? '/api/watchlist/add' : '/api/watchlist/remove';

  fetch(apiUrl, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      media_type: mediaType,
      item_id: itemId
    })
  })
  .then(response => response.json())
  .then(data => {
    if (data.success) {
      updateWatchlistButtonStates(mediaType, itemId, action, isCardButton, buttonElement);
      const message = (action === 'add') 
        ? 'İçerik izleme listenize eklendi!' 
        : 'İçerik izleme listenizden çıkarıldı!';
      showNotification(message);

      // Öneri sayfasındaysak ve bir şey eklendiyse önerileri yenile
      if (action === 'add' && window.location.pathname.includes('/recommendations')) {
        // refreshRecommendations(); // Bu fonksiyonu daha sonra iyileştirebiliriz
      }
      // İzleme listesi sayfasındaysak ve bir şey çıkarıldıysa kartı DOM'dan kaldır
      if (action === 'remove' && window.location.pathname.includes('/watchlist') && isCardButton) {
        const card = buttonElement.closest('.card');
        if (card) {
            card.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
            card.style.opacity = '0';
            card.style.transform = 'scale(0.95)';
            setTimeout(() => card.remove(), 300);
        }
      }

    } else {
      showNotification(data.error || 'Bir hata oluştu, lütfen tekrar deneyin.', 'error');
    }
  })
  .catch(error => {
    console.error('Watchlist Hata:', error);
    showNotification('Bir hata oluştu, lütfen tekrar deneyin.', 'error');
  });
}

// Buton ve etiket durumlarını güncelle
function updateWatchlistButtonStates(mediaType, itemId, action, isCardButton, clickedButton) {
    const cards = document.querySelectorAll(`.card`);
    cards.forEach(card => {
        const cardItemId = parseInt(card.querySelector('[data-item-id]')?.dataset.itemId);
        const cardMediaType = card.querySelector('[data-media-type]')?.dataset.mediaType;

        if (cardItemId === itemId && cardMediaType === mediaType) {
            const addBtn = card.querySelector('.add-to-watchlist-card');
            const removeBtn = card.querySelector('.remove-from-watchlist-card');
            let watchlistTag = card.querySelector('.watchlist-tag');

            if (action === 'add') {
                card.classList.add('in-watchlist');
                if (addBtn) addBtn.classList.add('hidden');
                if (removeBtn) removeBtn.classList.remove('hidden');
                if (!watchlistTag) {
                    watchlistTag = document.createElement('div');
                    watchlistTag.className = 'watchlist-tag';
                    watchlistTag.innerHTML = '<i class="fas fa-check"></i> İzleme Listemde';
                    // Etiketi card-image-wrapper'ın içine ekleyebiliriz ya da direkt card'ın başına
                    const imageWrapper = card.querySelector('.card-image-wrapper');
                    if (imageWrapper) {
                        imageWrapper.insertBefore(watchlistTag, imageWrapper.firstChild);
                    } else {
                        card.insertBefore(watchlistTag, card.firstChild);
                    }
                }
                setTimeout(() => { // CSS animasyonunun tamamlanması için küçük bir gecikme
                    if(watchlistTag) watchlistTag.style.opacity = '1';
                    if(watchlistTag) watchlistTag.style.transform = 'translateY(0)';
                }, 50);

            } else { // remove action
                card.classList.remove('in-watchlist');
                if (addBtn) addBtn.classList.remove('hidden');
                if (removeBtn) removeBtn.classList.add('hidden');
                if (watchlistTag) {
                    watchlistTag.style.opacity = '0';
                    watchlistTag.style.transform = 'translateY(-10px)';
                    setTimeout(() => watchlistTag.remove(), 300); 
                }
            }
        }
    });

    // Detay sayfasındaki butonları da güncelle (eğer oradaysak ve tıklanan buton kart butonu değilse)
    if (!isCardButton) {
        const detailAddBtn = document.getElementById('add-to-watchlist');
        const detailRemoveBtn = document.getElementById('remove-from-watchlist');
        if (detailAddBtn && detailRemoveBtn) {
            if (action === 'add') {
                detailAddBtn.classList.add('hidden');
                detailRemoveBtn.classList.remove('hidden');
            } else {
                detailAddBtn.classList.remove('hidden');
                detailRemoveBtn.classList.add('hidden');
            }
        }
    }
}

// Derecelendirme sistemi
function setupRatingSystem() {
  const ratingContainer = document.querySelector('.rating');
  if (ratingContainer) {
    const mediaType = ratingContainer.dataset.mediaType;
    const itemId = parseInt(ratingContainer.dataset.itemId, 10);
    const ratingInputs = document.querySelectorAll('.rating input');
    
    ratingInputs.forEach(input => {
      input.addEventListener('change', function() {
        const rating = parseFloat(this.value);
        
        fetch('/api/rate', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            media_type: mediaType,
            item_id: itemId,
            rating: rating
          })
        })
        .then(response => response.json())
        .then(data => {
          if (data.success) {
            showNotification(`${rating} yıldız değerlendirmeniz kaydedildi!`);
          }
        })
        .catch(error => {
          console.error('Hata:', error);
          showNotification('Değerlendirme kaydedilemedi, lütfen tekrar deneyin.', 'error');
        });
      });
    });
  }
}

// Tab yönetimi (Güncellenmiş)
function setupTabs(tabGroupSelector, tabSelector, contentItemSelector, activeTabClass = 'active', hiddenContentClass = 'hidden') {
  const tabGroups = document.querySelectorAll(tabGroupSelector);

  tabGroups.forEach(group => {
    const tabs = group.querySelectorAll(tabSelector);
    if (tabs.length === 0) return; // Bu grupta tab yoksa devam etme

    // Bu tab grubuna ait tüm hedef içerik ID'lerini topla
    const targetIds = Array.from(tabs)
      .map(tab => tab.dataset.target)
      .filter(id => !!id);

    // ID'lere karşılık gelen içerik elemanlarını DOM'dan al
    const contentItems = targetIds
      .map(id => document.getElementById(id))
      .filter(el => !!el);

    // Helper: belirtilen içerik öğesini aktif hale getir, diğerlerini gizle
    const activateContent = targetId => {
      contentItems.forEach(content => {
        if (content.id === targetId) {
          content.classList.remove(hiddenContentClass);
          content.classList.add(activeTabClass);
        } else {
          content.classList.add(hiddenContentClass);
          content.classList.remove(activeTabClass);
        }
      });
    };

    // İlk yüklendiğinde aktif tabı ve içeriği ayarla
    let initialActiveTab = group.querySelector(`${tabSelector}.${activeTabClass}`);
    if (!initialActiveTab) {
      // Eğer template tarafında bir aktif tab işaretlenmemişse ilk tabı aktif yap
      initialActiveTab = tabs[0];
      initialActiveTab.classList.add(activeTabClass);
    }
    activateContent(initialActiveTab.dataset.target);

    // Tab tıklama olaylarını bağla
    tabs.forEach(tab => {
      tab.addEventListener('click', function (event) {
        event.preventDefault();

        // Tüm tablardan aktif sınıfını kaldır
        tabs.forEach(t => t.classList.remove(activeTabClass));
        // Bu tabı aktif yap
        this.classList.add(activeTabClass);

        // İlgili içeriği göster
        const targetId = this.dataset.target;
        activateContent(targetId);
      });
    });
  });
}

// Bildirim gösterme
function showNotification(message, type = 'success') {
  // Mevcut bildirimleri kaldır
  const existingNotifications = document.querySelectorAll('.notification');
  existingNotifications.forEach(notification => {
    notification.remove();
  });
  
  // Yeni bildirim oluştur
  const notification = document.createElement('div');
  notification.className = `notification notification-${type}`;
  notification.textContent = message;
  
  // Bildirim konteyneri oluştur (yoksa)
  let notificationContainer = document.querySelector('.notification-container');
  if (!notificationContainer) {
    notificationContainer = document.createElement('div');
    notificationContainer.className = 'notification-container';
    document.body.appendChild(notificationContainer);
  }
  
  // Bildirimi ekle
  notificationContainer.appendChild(notification);
  
  // Bildirimi otomatik kapat
  setTimeout(() => {
    notification.classList.add('notification-hide');
    setTimeout(() => {
      notification.remove();
    }, 300);
  }, 3000);
}

// Arama formu gönderimi
document.addEventListener('DOMContentLoaded', function() {
  const searchForm = document.querySelector('.search-form');
  if (searchForm) {
    searchForm.addEventListener('submit', function(e) {
      e.preventDefault();
      
      const searchInput = document.querySelector('.search-input').value.trim();
      if (searchInput === '') {
        return;
      }
      
      // Medya türünü al
      let mediaType = 'movie';
      const mediaTypeRadios = document.querySelectorAll('input[name="media_type"]');
      mediaTypeRadios.forEach(radio => {
        if (radio.checked) {
          mediaType = radio.value;
        }
      });
      
      // Arama sayfasına yönlendir
      window.location.href = `/search?q=${encodeURIComponent(searchInput)}&type=${mediaType}`;
    });
  }
});

// Hero Section için Parıltı Efekti (Global olarak tanımlı kalsın, çağrımı SPA'dan)
function createHeroSparkles() {
  const sparkleContainer = document.querySelector('.hero-sparkle-container');
  if (!sparkleContainer) return;

  const numberOfSparkles = 50; // Ekranda aynı anda görünecek maksimum parıltı sayısı
  let activeSparkles = 0;

  function createSparkle() {
    if (activeSparkles >= numberOfSparkles) return;

    const sparkle = document.createElement('div');
    sparkle.classList.add('hero-sparkle');

    // Rastgele konum
    const x = Math.random() * 100; // Yüzde olarak konum
    const y = Math.random() * 100;
    sparkle.style.left = `${x}%`;
    sparkle.style.top = `${y}%`;

    // Rastgele boyut ve animasyon süresi
    const size = Math.random() * 3 + 1; // 1px ile 4px arası boyut
    sparkle.style.width = `${size}px`;
    sparkle.style.height = `${size}px`;
    const duration = Math.random() * 2 + 1; // 1sn ile 3sn arası animasyon

    sparkleContainer.appendChild(sparkle);
    activeSparkles++;

    // Animasyonu başlat (CSS'te tanımlanacak bir animasyon adı varsayılıyor)
    // Şimdilik basit bir opacity animasyonu yapalım, CSS'te daha karmaşık hale getirilebilir.
    sparkle.animate([
      { opacity: 0, transform: 'scale(0.5)' },
      { opacity: 1, transform: 'scale(1)' },
      { opacity: 0, transform: 'scale(0.5)' }
    ], {
      duration: duration * 1000, // Milisaniye cinsinden
      easing: 'ease-in-out'
    });

    // Animasyon bitince DOM'dan kaldır ve sayacı düşür
    setTimeout(() => {
      sparkle.remove();
      activeSparkles--;
    }, duration * 1000);
  }

  // Belirli aralıklarla yeni parıltılar oluştur
  setInterval(createSparkle, 200); // Her 200ms'de bir parıltı oluşturmayı dene
}

// Tam Sayfa Kaydırma (Global olarak tanımlı kalsın, çağrımı SPA'dan)
function setupFullpageScroll() {
  const container = document.getElementById('fullpage-container');
  const sections = document.querySelectorAll('.fullpage-section'); // Bu, dinamik olarak değişebilir
  const body = document.body; // body class'ını kontrol etmek için

  if (!container || sections.length === 0 || !body.classList.contains('page-index')) return;

  let currentSection = 0;
  let isScrolling = false;
  const scrollThreshold = 50; 
  let touchStartY = 0;

  // updateDotNavActiveState burada veya scrollToSection içinde çağrılabilir
  updateDotNavActiveState(currentSection); 

  container.addEventListener('wheel', function(event) {
    // ... (passive: false ile preventDefault)
    if (!body.classList.contains('page-index')) return; // Sadece ana sayfada çalışsın
    event.preventDefault(); 
    if (isScrolling) return;

    if (event.deltaY > scrollThreshold) {
      scrollToSection(currentSection + 1);
    } else if (event.deltaY < -scrollThreshold) {
      scrollToSection(currentSection - 1);
    }
  }, { passive: false });

  container.addEventListener('touchstart', function(event) {
    if (!body.classList.contains('page-index')) return;
    touchStartY = event.touches[0].clientY;
  }, { passive: true });

  container.addEventListener('touchmove', function(event) {
    if (!body.classList.contains('page-index') || isScrolling) return;
    const touchEndY = event.touches[0].clientY;
    const deltaY = touchStartY - touchEndY; 

    if (deltaY > scrollThreshold) {
      scrollToSection(currentSection + 1);
    } else if (deltaY < -scrollThreshold) {
      scrollToSection(currentSection - 1);
    }
  }, { passive: true });


  function scrollToSection(sectionIndex) {
    console.log('scrollToSection çağrıldı, sectionIndex:', sectionIndex, 'isScrolling:', isScrolling); // DEBUG
    const currentSections = container.querySelectorAll('.fullpage-section'); // Her zaman güncel listeyi al
    if (isScrolling && sectionIndex === currentSection) return; 
    if (sectionIndex >= 0 && sectionIndex < currentSections.length) {
      isScrolling = true;
      currentSections[sectionIndex].scrollIntoView({ behavior: 'smooth' });
      currentSection = sectionIndex;
      updateDotNavActiveState(currentSection); 

      // Dinamik içerik yükleme (SADECE ANA SAYFADA)
      if (body.classList.contains('page-index')) {
        const targetSectionElement = currentSections[sectionIndex];
        if (sectionIndex === 1 && targetSectionElement.id === 'section-watchlist-content' && !targetSectionElement.dataset.loaded) {
          fetchAndInjectSection('/watchlist', targetSectionElement);
        } else if (sectionIndex === 2 && targetSectionElement.id === 'section-recommendations-content' && !targetSectionElement.dataset.loaded) {
          fetchAndInjectSection('/recommendations', targetSectionElement);
        }
      }
      
      setTimeout(() => {
        isScrolling = false;
      }, 700); 
    } else {
      isScrolling = false; 
    }
  }
  // updateDotNavActiveState, scrollToSection içinde çağrılıyor.
  // Yatay kaydırma için olan wheel listener KALACAK
  const horizontalWrappers = document.querySelectorAll('.horizontal-scroll-wrapper');
  horizontalWrappers.forEach(wrapper => {
    wrapper.addEventListener('wheel', function(event) {
      if (wrapper.scrollWidth > wrapper.clientWidth) {
        if (Math.abs(event.deltaX) > Math.abs(event.deltaY)) {
             return;
        }
      }
    }, { passive: true }); 
  });
}

// Yeni Dinamik İçerik Yükleme Fonksiyonu
function fetchAndInjectSection(url, targetSectionElement) {
    console.log('fetchAndInjectSection çağrıldı, url:', url, 'loaded:', targetSectionElement.dataset.loaded); // DEBUG
    const placeholderContainer = targetSectionElement.querySelector('.placeholder-container');
    if (placeholderContainer) {
        placeholderContainer.innerHTML = `<i class="fas fa-spinner fa-spin"></i> <p>${targetSectionElement.dataset.title} yükleniyor...</p>`;
    }

    fetch(url)
        .then(response => response.text())
        .then(html => {
            const parser = new DOMParser();
            const doc = parser.parseFromString(html, 'text/html');
            // Diğer sayfalardaki içerik her zaman main-content-wrapper içinde olmalı
            const newContent = doc.getElementById('main-content-wrapper')?.innerHTML;

            if (newContent) {
                // Yer tutucuyu kaldırıp yeni içeriği ekle
                targetSectionElement.innerHTML = newContent; 
                targetSectionElement.dataset.loaded = 'true';

                // Yüklenen içerik için gerekli JS'leri yeniden başlat
                setupWatchlistButtons(); 
                setupRatingSystem();   
                setupTabs('.tabs', '.tab', '.tab-content'); 
            } else {
                if (placeholderContainer) placeholderContainer.innerHTML = `<p>${targetSectionElement.dataset.title} içeriği yüklenemedi.</p>`;
                console.error(`'main-content-wrapper' içeriği bulunamadı: ${url}`);
            }
        })
        .catch(error => {
            console.error('Dinamik bölüm yükleme hatası:', error);
            if (placeholderContainer) placeholderContainer.innerHTML = `<p>${targetSectionElement.dataset.title} yüklenirken bir hata oluştu.</p>`;
        });
}


function initSpaNavigation() {
    const mainContentWrapper = document.getElementById('main-content-wrapper');
    // const fullpageContainer = document.getElementById('fullpage-container'); // Artık doğrudan yönetilmiyor
    const navLinks = document.querySelectorAll('a.nav-item:not(#clearAllDataBtn), a.overlay-nav-item:not(#clearAllDataBtnOverlay)'); // Tüm navigasyon linklerini hedefle
    const body = document.body;

    if (!mainContentWrapper) {
        console.error('SPA Navigation için #main-content-wrapper bulunamadı!');
        return;
    }

    handlePageStructure(window.location.pathname); // Sayfa ilk yüklendiğinde doğru durumu ayarla

    document.body.addEventListener('click', function(event) {
        let target = event.target;
        // Linkin kendisi veya içindeki bir element (örn. <i>) tıklanmış olabilir.
        while (target && target !== this && target.tagName !== 'A') {
            target = target.parentElement;
        }

        if (target && target.matches('a.nav-item:not(#clearAllDataBtn), a.overlay-nav-item:not(#clearAllDataBtnOverlay)')) {
            event.preventDefault();
            const url = target.href;
            const title = (target.innerText || target.textContent || target.dataset.title || 'MovieApp').trim() + " - MovieApp";

            // Menüdeki aktif sınıfı yönet (hem header hem overlay için)
            document.querySelectorAll('a.nav-item, a.overlay-nav-item').forEach(item => item.classList.remove('active'));
            target.classList.add('active');
            // Eğer overlay menüde ise, headerdaki eşleniğini de aktif yap (veya tam tersi)
            const targetHref = target.getAttribute('href');
            document.querySelectorAll(`a.nav-item[href="${targetHref}"], a.overlay-nav-item[href="${targetHref}"]`).forEach(link => link.classList.add('active'));
            
            // Hamburger menüsü açıksa kapat
            const overlayNav = document.getElementById('overlay-nav');
            if (overlayNav && overlayNav.classList.contains('open')) {
                toggleOverlayMenu(false); // setupHamburgerMenu içinde tanımlı olmalı
            }

            loadPageContent(url, title);
        }
    });

    window.addEventListener('popstate', function(event) {
        const path = window.location.pathname;
        const title = (event.state && event.state.title) || document.title;
        loadPageContent(window.location.href, title, false);
    });

    function handlePageStructure(pathname) {
        const isHomePage = pathname === '/';
        if (isHomePage) {
            body.className = 'page-index fullpage-active';
        } else {
            body.className = 'page-standard';
        }
        // JS fonksiyon çağrıları loadPageContent sonrasına taşındı
    }

    function loadPageContent(url, title, addToHistory = true) {
        mainContentWrapper.classList.add('page-content-exit'); // Çıkış animasyonunu başlat

        fetch(url)
            .then(response => response.text())
            .then(html => {
                setTimeout(() => { // Animasyon süresi kadar bekle
                    mainContentWrapper.classList.remove('page-content-exit');
                    mainContentWrapper.innerHTML = ''; // Önce temizle

                    const parser = new DOMParser();
                    const doc = parser.parseFromString(html, 'text/html');
                    const newPageTitle = doc.querySelector('title')?.textContent || title;
                    document.title = newPageTitle;

                    // Yeni içeriği yerleştir (her zaman mainContentWrapper içine)
                    const newMainContent = doc.getElementById('main-content-wrapper')?.innerHTML;
                    if (newMainContent) {
                        mainContentWrapper.innerHTML = newMainContent;
                    } else {
                        // Eğer gelen HTML'de main-content-wrapper yoksa, belki de doğrudan body içeriğidir (index.html gibi)
                        // Ya da block content'i arayabiliriz.
                        // Şimdilik, eğer main-content-wrapper yoksa tüm body'yi almaya çalışmak yerine hata verelim.
                        mainContentWrapper.innerHTML = '<div class="container text-center" style="padding: 50px;"><h2>Sayfa içeriği formatı hatalı.</h2></div>';
                        console.error("'main-content-wrapper' ID'li element kaynak HTML'de bulunamadı: ", url);
                        // return; // Hata durumunda işlemi durdurabiliriz
                    }

                    if (addToHistory) {
                        history.pushState({ url: url, title: newPageTitle }, newPageTitle, url);
                    }
                    
                    // Sayfa tipine göre body class'ını ve JS'leri ayarla
                    const isTargetHomePage = (new URL(url)).pathname === '/';
                    handlePageStructure((new URL(url)).pathname); // body class'ını ayarla

                    // Her zaman çalışacak temel JS başlatıcıları
                    setupWatchlistButtons();
                    setupRatingSystem();
                    setupTabs('.tabs', '.tab', '.tab-content');
                    // setupClearAllDataButton(); // Bu DOM yüklendiğinde bir kere çalışsa yeterli
                    // setupHamburgerMenu(); // Bu da DOM yüklendiğinde bir kere çalışsa yeterli

                    if (isTargetHomePage) {
                        createHeroSparkles();
                        setupFullpageScroll(); // Bu, içindeki section'ları bulup dotnav'ı güncelleyecektir
                        setupDotNavigation(); // Nokta navigasyon eventlerini bağla
                    } else {
                        // Diğer sayfalarda genel tablar varsa çalıştır
                        // Veya her sayfa kendi özel tab setup'ını yapsın.
                        // Şimdilik tüm sayfalarda genel .tab/.tab-content için çalışsın.
                        setupTabs('.tabs', '.tab', '.tab-content'); 
                    }
                    
                    // ÖNERİ SAYFASINA ÖZEL TAB KURULUMU (SPA ile gelindiğinde)
                    const currentPath = (new URL(url)).pathname;
                    if (currentPath.includes('/recommendations')) {
                        const newDoc = mainContentWrapper; // Artık parse edilmiş HTML bu wrapper'da
                        if (newDoc.querySelector('.ml-tabs')) {
                            console.log("ML tabs found via SPA, setting them up...");
                            setupTabs('.ml-tabs', '.recommendations-tab', '.recommendations-content');
                        }
                        if (newDoc.querySelector('.tmdb-tabs')) {
                            console.log("TMDB tabs found via SPA, setting them up...");
                            setupTabs('.tmdb-tabs', '.recommendations-tab', '.recommendations-content');
                        }
                    }

                    const dotNav = document.querySelector('.dot-nav');
                    if (dotNav) {
                        dotNav.style.display = isTargetHomePage ? 'block' : 'none';
                    }
                    
                    mainContentWrapper.classList.add('page-content-enter');
                    requestAnimationFrame(() => { mainContentWrapper.classList.add('page-content-enter-active'); });
                    setTimeout(() => mainContentWrapper.classList.remove('page-content-enter', 'page-content-enter-active'), 500);
                    
                    window.scrollTo(0, 0);
                }, 300); // Çıkış animasyon süresi
            })
            .catch(error => {
                console.error('Sayfa yükleme hatası:', error);
                mainContentWrapper.classList.remove('page-content-exit');
                mainContentWrapper.innerHTML = '<div class="container text-center" style="padding: 50px;"><h2>Sayfa yüklenirken bir hata oluştu.</h2><p>Lütfen daha sonra tekrar deneyin.</p></div>';
            });
    }
}

// toggleOverlayMenu fonksiyonu setupHamburgerMenu içinde tanımlı olmalı
// Eğer global ise, setupHamburgerMenu'dan önce veya sonra tanımlanabilir.
// Şimdilik setupHamburgerMenu'nun bunu global yapacağını varsayalım veya oraya taşıyalım.
let toggleOverlayMenu = function(forceOpen) { /* placeholder */ };

function setupHamburgerMenu() {
    const hamburgerBtn = document.getElementById('hamburger-menu-button');
    const overlayNav = document.getElementById('overlay-nav');
    const overlayNavCloseBtn = document.getElementById('overlay-nav-close');
    // const overlayNavItems = document.querySelectorAll('.overlay-nav-item'); // Artık initSpaNavigation'da yönetiliyor

    if (!hamburgerBtn || !overlayNav || !overlayNavCloseBtn) return;

    // Global toggleOverlayMenu'yu burada tanımla
    toggleOverlayMenu = function(forceOpen) {
        const isOpen = overlayNav.classList.contains('open');
        if (typeof forceOpen === 'boolean' ? forceOpen : !isOpen) {
            overlayNav.classList.add('open');
            hamburgerBtn.classList.add('active');
            document.body.style.overflow = 'hidden'; 
            hamburgerBtn.setAttribute('aria-expanded', 'true');
        } else {
            overlayNav.classList.remove('open');
            hamburgerBtn.classList.remove('active');
            if (!document.body.classList.contains('fullpage-active')) {
                 document.body.style.overflowY = 'auto';
            }
            hamburgerBtn.setAttribute('aria-expanded', 'false');
        }
    }

    hamburgerBtn.addEventListener('click', () => {
        toggleOverlayMenu();
    });

    overlayNavCloseBtn.addEventListener('click', () => {
        toggleOverlayMenu(false);
    });

    // Overlay menü içindeki temizle butonu için özel event listener
    const clearAllDataBtnOverlay = document.getElementById('clearAllDataBtnOverlay');
    if(clearAllDataBtnOverlay) {
        clearAllDataBtnOverlay.addEventListener('click', () => {
            const mainClearBtn = document.getElementById('clearAllDataBtn'); // base.html'deki ana buton
            if(mainClearBtn) mainClearBtn.click(); 
            toggleOverlayMenu(false);
        });
    }
}

// Nokta Navigasyonu (Sadece ana sayfada çalışır, eventleri ayarlar)
function setupDotNavigation() {
    const dotNav = document.querySelector('.dot-nav');
    if (!document.body.classList.contains('page-index') || !dotNav) {
        if(dotNav) dotNav.style.display = 'none'; // Ana sayfada değilse gizle
        return;
    }
    if(dotNav) dotNav.style.display = 'block'; // Ana sayfada ise göster

    const dots = dotNav.querySelectorAll('a');
    const container = document.getElementById('fullpage-container'); // Kaydırılacak container

    if (!container || dots.length === 0) return;

    dots.forEach(dot => {
        dot.addEventListener('click', function(event) {
            event.preventDefault();
            const sectionId = this.getAttribute('href'); // #section-id
            const targetSection = document.querySelector(sectionId);
            if (targetSection) {
                targetSection.scrollIntoView({ behavior: 'smooth' });
                // Aktif bölüm ve nokta güncellemesi scrollToSection içinde yapılmalı
            }
        });
    });
}

// updateDotNavActiveState fonksiyonu (setupFullpageScroll içinde de referans veriliyor)
function updateDotNavActiveState(index) {
    const dots = document.querySelectorAll('.dot-nav a');
    dots.forEach((dot, i) => {
        if (i === index) {
            dot.classList.add('active');
        } else {
            dot.classList.remove('active');
        }
    });
}

// Bölüm görünürlüğüne göre dinamik içerik yükleyici
function setupDynamicSectionLoader() {
  const container = document.getElementById('fullpage-container');
  if (!container) return;

  const observerOptions = {
    root: container,
    threshold: 0.4 // Bölümün %40'ı görünür olunca tetikle
  };

  const sectionObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const sectionEl = entry.target;
        if (!sectionEl.dataset.loaded && sectionEl.dataset.url) {
          // İlk kez görünüyorsa yükle
          fetchAndInjectSection(sectionEl.dataset.url, sectionEl);
        }
      }
    });
  }, observerOptions);

  // İzleme listesi ve öneriler bölümlerini gözlemle
  const dynamicSections = document.querySelectorAll('#section-watchlist-content, #section-recommendations-content');
  dynamicSections.forEach(sec => sectionObserver.observe(sec));
} 