"""Generates and audits the 200-example Golden Evaluation Set with strict taxonomy adherence."""
import os
import sys
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.utils.io import load_yaml
from src.utils.logger import setup_logger

logger = setup_logger("generate_golden_set")

GOLDEN_ITEMS = [
    # 1. Battery & Power Management (20 items)
    ("battery_power_issue", "My iPhone 7 battery is draining 1% every two minutes since updating to iOS 11. Can you help?", "auto", "Standard battery drain troubleshooting link and settings check.", "easy", "conv_gold_001"),
    ("battery_power_issue", "Phone gets burning hot while plugged into the original wall charger and shuts down.", "escalate", "Safety hazard / potential hardware thermal overload.", "high_risk", "conv_gold_002"),
    ("battery_power_issue", "Why does my battery jump from 40% straight to 1% and then die?", "auto", "Battery calibration / battery health advisory.", "medium", "conv_gold_003"),
    ("battery_power_issue", "My iPad Pro won't charge past 80% no matter how long it stays plugged in.", "auto", "Optimized battery charging guidance.", "easy", "conv_gold_004"),
    ("battery_power_issue", "Battery health says 74% and Service recommended. How do I get it replaced?", "auto", "Battery replacement service booking details.", "easy", "conv_gold_005"),
    ("battery_power_issue", "Phone dies at 30% when I'm outside in cold weather.", "auto", "Cold weather lithium-ion operating temperature guidance.", "medium", "conv_gold_006"),
    ("battery_power_issue", "The back of my iPhone is swelling and pushing the screen out from the frame!", "escalate", "Swollen battery physical safety risk; immediate hardware service.", "high_risk", "conv_gold_007"),
    ("battery_power_issue", "Is fast charging supported on the iPhone 8 with standard iPad 12W brick?", "auto", "Power adapter compatibility specifications.", "easy", "conv_gold_008"),
    ("battery_power_issue", "My iPhone battery indicator is stuck on 100% all day even after heavy usage.", "auto", "Software reboot / battery sensor reset instructions.", "medium", "conv_gold_009"),
    ("battery_power_issue", "Charger cable sparked and now the lightning port smells like smoke.", "escalate", "Electrical/fire hazard requiring urgent physical hardware inspection.", "high_risk", "conv_gold_010"),
    ("battery_power_issue", "Can you check why Settings > Battery says 'Background Activity' took 60% battery for YouTube?", "auto", "Background app refresh settings guidance.", "easy", "conv_gold_011"),
    ("battery_power_issue", "Does low power mode turn off automatic mail fetch?", "auto", "Low Power Mode feature explanation.", "easy", "conv_gold_012"),
    ("battery_power_issue", "My phone won't turn on at all, black screen, tried charging for 3 hours with 3 different cables.", "escalate", "Complete power failure requiring hardware diagnostic or Genius Bar.", "hard", "conv_gold_013"),
    ("battery_power_issue", "Is it normal for a brand new iPhone X to lose 10% overnight on standby?", "auto", "Standby power optimization tips.", "easy", "conv_gold_014"),
    ("battery_power_issue", "Battery widget is not showing battery level for my connected Bluetooth headphones.", "auto", "Widget troubleshooting / Bluetooth device pairing reset.", "medium", "conv_gold_015"),
    ("battery_power_issue", "How many charge cycles before iPhone battery capacity drops below 80%?", "auto", "Battery lifecycle technical specification.", "easy", "conv_gold_016"),
    ("battery_power_issue", "Wireless charging pad keeps stopping and flashing blue after 5 minutes.", "auto", "Qi wireless charging alignment and case thickness advice.", "medium", "conv_gold_017"),
    ("battery_power_issue", "My phone is dropping battery fast and my screen is also flickering green lines.", "escalate", "Dual symptom: battery + hardware display failure.", "hard", "conv_gold_018"),
    ("battery_power_issue", "Is there a recall program for iPhone 6s unexpected shutdown issues?", "auto", "Apple Quality Program serial number check advisory.", "medium", "conv_gold_019"),
    ("battery_power_issue", "Turned on Low Power Mode and my phone feels extremely laggy and slow.", "auto", "CPU throttling explanation under Low Power Mode.", "easy", "conv_gold_020"),

    # 2. Software & OS Update Problems (20 items)
    ("software_update_issue", "My iPhone is stuck on the Apple logo with progress bar for 4 hours while updating to iOS 11.", "auto", "Force restart / Recovery Mode iTunes restore steps.", "easy", "conv_gold_021"),
    ("software_update_issue", "How do I downgrade my iPhone from iOS 11 back to iOS 10.3.3?", "auto", "Official unsigned iOS downgrade policy explanation.", "medium", "conv_gold_022"),
    ("software_update_issue", "Unable to Verify Update. iOS failed verification because you are no longer connected to internet.", "auto", "Network settings reset and re-download update file steps.", "easy", "conv_gold_023"),
    ("software_update_issue", "Update requested... has been spinning on my screen for two days without downloading.", "auto", "Delete downloaded OTA installer and restart download.", "easy", "conv_gold_024"),
    ("software_update_issue", "Error 4013 occurred while trying to restore iPhone in iTunes.", "auto", "iTunes error code 4013 troubleshooting (cable, USB port, OS update).", "medium", "conv_gold_025"),
    ("software_update_issue", "My phone went into a continuous boot loop after latest update and I have unsaved medical data.", "escalate", "Data loss risk with critical medical information requiring specialist.", "high_risk", "conv_gold_026"),
    ("software_update_issue", "When is iOS 11.1 coming out to fix the calculator typing bug?", "auto", "Software update release schedule policy.", "easy", "conv_gold_027"),
    ("software_update_issue", "Settings app has red badge '1' for software update but when I tap it says software is up to date.", "auto", "Notification cache refresh / restart steps.", "medium", "conv_gold_028"),
    ("software_update_issue", "iTunes says 'The iPhone could not be restored. An unknown error occurred (9).'", "auto", "Security software / USB cable troubleshooting for error 9.", "medium", "conv_gold_029"),
    ("software_update_issue", "Is iPhone 5s supported on iOS 11 or will it slow it down too much?", "auto", "Device compatibility specs for iOS 11.", "easy", "conv_gold_030"),
    ("software_update_issue", "After updating, all my default apps like Weather and Stocks are missing.", "auto", "Re-downloading built-in apps from App Store.", "medium", "conv_gold_031"),
    ("software_update_issue", "Can I update my iPad over cellular data or do I strictly need Wi-Fi?", "auto", "Cellular download size limit policy and iTunes alternative.", "easy", "conv_gold_032"),
    ("software_update_issue", "Software update says 'Not enough storage space available' even after deleting 5GB of videos.", "auto", "Manage storage 'Other' category and iTunes update method.", "medium", "conv_gold_033"),
    ("software_update_issue", "Updated my Mac to High Sierra and now it gives a Kernel Panic kernel error on boot.", "escalate", "macOS kernel panic and crash dump analysis requiring tier-2 support.", "hard", "conv_gold_034"),
    ("software_update_issue", "Will updating my phone wipe all my text messages and photos?", "auto", "Standard update vs restore data retention explanation.", "easy", "conv_gold_035"),
    ("software_update_issue", "Installed public beta profile, how do I remove beta profile to receive public release?", "auto", "Profile removal in Settings > General > VPN & Device Management.", "easy", "conv_gold_036"),
    ("software_update_issue", "Why does iOS update keep pausing whenever my screen locks?", "auto", "Keep connected to power and active Wi-Fi guidance.", "easy", "conv_gold_037"),
    ("software_update_issue", "Can you send me the IPSW file directly on Twitter?", "auto", "Security guideline: Apple does not distribute raw IPSW via DM.", "medium", "conv_gold_038"),
    ("software_update_issue", "Software update bricked my device completely, it won't even show battery icon or charge.", "escalate", "Bricked device requiring hardware replacement evaluation.", "hard", "conv_gold_039"),
    ("software_update_issue", "My 3D Touch feels different and haptic feedback is delayed since updating.", "auto", "3D touch sensitivity settings adjustment steps.", "medium", "conv_gold_040"),

    # 3. Apple ID, Security & Authentication (20 items)
    ("account_security_issue", "My Apple ID is locked for security reasons and I can't receive verification SMS codes.", "escalate", "Account security lock requiring identity verification via iforgot.apple.com.", "high_risk", "conv_gold_041"),
    ("account_security_issue", "I got an email saying my Apple ID was logged into from Moscow, Russia. I am in Ohio!", "escalate", "Compromised account security / unauthorized access alert.", "high_risk", "conv_gold_042"),
    ("account_security_issue", "How do I turn on Two-Factor Authentication on my iPhone?", "auto", "Standard 2FA setup steps in Settings > Password & Security.", "easy", "conv_gold_043"),
    ("account_security_issue", "Forgot my Apple ID password and lost access to my trusted phone number.", "escalate", "Account recovery process required with human identity check.", "hard", "conv_gold_044"),
    ("account_security_issue", "Someone hacked my iCloud and is threatening to wipe my phone unless I pay them Bitcoin!", "escalate", "Extortion / criminal security threat requiring immediate fraud escalation.", "high_risk", "conv_gold_045"),
    ("account_security_issue", "How do I remove an old device from my trusted devices list?", "auto", "Settings > Apple ID > Device list removal steps.", "easy", "conv_gold_046"),
    ("account_security_issue", "I bought a used iPhone on Craigslist and it has Activation Lock with someone else's email.", "auto", "Activation Lock policy: original receipt required or contact previous owner.", "medium", "conv_gold_047"),
    ("account_security_issue", "Can Apple Support disable Activation Lock if I give you the IMEI number here on Twitter?", "auto", "Security policy: Activation Lock cannot be bypassed over Twitter.", "easy", "conv_gold_048"),
    ("account_security_issue", "Two-factor code is being sent to my stolen phone. How can I get into my account?", "escalate", "Stolen device and 2FA recovery requiring account recovery protocol.", "high_risk", "conv_gold_049"),
    ("account_security_issue", "How do I generate an app-specific password for third-party email client?", "auto", "appleid.apple.com app-specific password generation guidance.", "easy", "conv_gold_050"),
    ("account_security_issue", "My child changed the screen time passcode and now nobody knows it.", "auto", "Screen Time passcode reset via Apple ID credentials.", "medium", "conv_gold_051"),
    ("account_security_issue", "I'm receiving multiple 2FA verification prompts on my phone that I did not request!", "escalate", "Active account brute-force attack; prompt password change and escalation.", "high_risk", "conv_gold_052"),
    ("account_security_issue", "How do I change the primary email address on my Apple ID account?", "auto", "Official guide on updating Apple ID username.", "easy", "conv_gold_053"),
    ("account_security_issue", "What is the waiting period for Apple ID Account Recovery?", "auto", "Account recovery timeframe explanation and status check portal.", "medium", "conv_gold_054"),
    ("account_security_issue", "I suspect someone keylogged my Mac and stole my keychain passwords.", "escalate", "High-severity security breach requiring security team.", "high_risk", "conv_gold_055"),
    ("account_security_issue", "Can two family members share one Apple ID for App Store and have separate iClouds?", "auto", "Family Sharing setup advisory instead of sharing accounts.", "medium", "conv_gold_056"),
    ("account_security_issue", "Why does Apple ID ask security questions instead of SMS code?", "auto", "Legacy security questions vs 2FA migration guidance.", "easy", "conv_gold_057"),
    ("account_security_issue", "A phishing site took my Apple credentials 5 minutes ago, what do I do immediately?", "escalate", "Immediate credential theft requiring fast containment protocol.", "high_risk", "conv_gold_058"),
    ("account_security_issue", "How do I sign out of Apple ID if 'Sign Out is not available due to restrictions'?", "auto", "Turn off Screen Time Content & Privacy Restrictions first.", "medium", "conv_gold_059"),
    ("account_security_issue", "Can you tell me what email address is associated with serial number F2LXXXXX?", "auto", "Privacy policy: Apple Support cannot disclose account owner details.", "easy", "conv_gold_060"),

    # 4. Billing, Subscriptions & Refunds (20 items)
    ("billing_subscription_issue", "I was charged $9.99 on my credit card from 'ITUNES.COM/BILL' and don't know what it is for.", "auto", "Check purchase history and reportaproblem.apple.com guide.", "easy", "conv_gold_061"),
    ("billing_subscription_issue", "How do I request a refund for an accidental in-app gem purchase my kid made?", "auto", "reportaproblem.apple.com refund submission instructions.", "easy", "conv_gold_062"),
    ("billing_subscription_issue", "Cancel my Apple Music subscription right now and refund this month's charge.", "auto", "Manage Subscriptions settings steps and refund request portal.", "easy", "conv_gold_063"),
    ("billing_subscription_issue", "You charged me $500 for a fraudulent App Store game, my lawyer will contact you if not refunded!", "escalate", "Large fraudulent charge dispute + legal threat requiring billing supervisor.", "high_risk", "conv_gold_064"),
    ("billing_subscription_issue", "How long does it take for an Apple Store refund to show up on my debit card?", "auto", "Standard refund timeframe (up to 30 days depending on financial institution).", "easy", "conv_gold_065"),
    ("billing_subscription_issue", "Payment method declined error when trying to download a free app.", "auto", "Unpaid balance check in billing information instructions.", "medium", "conv_gold_066"),
    ("billing_subscription_issue", "How do I change my billing payment method to PayPal?", "auto", "Settings > Payment & Shipping management guide.", "easy", "conv_gold_067"),
    ("billing_subscription_issue", "I have an active subscription but the app is still telling me to upgrade to premium.", "auto", "Restore Purchases within third-party app guidance.", "medium", "conv_gold_068"),
    ("billing_subscription_issue", "Can I use Apple Gift Card balance to pay for iCloud monthly storage?", "auto", "Gift card eligible balance spending rules.", "easy", "conv_gold_069"),
    ("billing_subscription_issue", "I was double-billed for the same movie rental on iTunes.", "auto", "reportaproblem.apple.com duplicate charge dispute guide.", "easy", "conv_gold_070"),
    ("billing_subscription_issue", "Why did Apple charge a $1 pending authorization on my card?", "auto", "Explanation of temporary verification hold.", "easy", "conv_gold_071"),
    ("billing_subscription_issue", "I filed a chargeback with my bank and now my whole Apple ID is disabled.", "escalate", "Chargeback lockdown requiring human finance/dispute team review.", "hard", "conv_gold_072"),
    ("billing_subscription_issue", "Can I transfer my paid App Store purchases to a different Apple ID account?", "auto", "Purchases are non-transferable policy; suggest Family Sharing.", "medium", "conv_gold_073"),
    ("billing_subscription_issue", "How do I view receipts for all my past App Store transactions?", "auto", "View purchase history in Account Settings instructions.", "easy", "conv_gold_074"),
    ("billing_subscription_issue", "Free trial charged me immediately upon sign up without giving the free 7 days.", "auto", "Free trial eligibility rules and refund request link.", "medium", "conv_gold_075"),
    ("billing_subscription_issue", "I redeemed a $50 gift card but my Apple ID balance shows $0.", "escalate", "Gift card redemption error requiring code audit and proof of purchase.", "hard", "conv_gold_076"),
    ("billing_subscription_issue", "How do I turn off recurring auto-renewal for HBO on Apple TV?", "auto", "Subscriptions settings management guide.", "easy", "conv_gold_077"),
    ("billing_subscription_issue", "Why does sales tax apply to digital App Store purchases in my state?", "auto", "Tax compliance explanation for digital goods.", "easy", "conv_gold_078"),
    ("billing_subscription_issue", "Can you issue a refund for an order placed on the online Apple Store via DM?", "auto", "Direct user to online Apple Store order status portal.", "easy", "conv_gold_079"),
    ("billing_subscription_issue", "My subscription was cancelled but the app continues to charge my bank account every month.", "escalate", "Recurring unauthorized billing error requiring human billing specialist.", "hard", "conv_gold_080"),

    # 5. Hardware, Screen & Physical Repairs (20 items)
    ("hardware_repair_issue", "I dropped my iPhone 8 and the glass back shattered. How much does out-of-warranty repair cost?", "auto", "Apple repair pricing guide and Genius Bar appointment link.", "easy", "conv_gold_081"),
    ("hardware_repair_issue", "The Home button on my iPhone 7 feels completely dead and doesn't click at all.", "auto", "AssistiveTouch temporary workaround + service options.", "medium", "conv_gold_082"),
    ("hardware_repair_issue", "My phone fell into a lake, and now the camera lens has water droplets inside.", "escalate", "Liquid ingress damage requiring in-person hardware evaluation.", "hard", "conv_gold_083"),
    ("hardware_repair_issue", "How do I book a Genius Bar appointment at the Fifth Avenue store in New York?", "auto", "Apple Support app / getsupport.apple.com booking link.", "easy", "conv_gold_084"),
    ("hardware_repair_issue", "Touch screen has ghost touch and opens random apps by itself.", "auto", "Display replacement / multi-touch service program check.", "medium", "conv_gold_085"),
    ("hardware_repair_issue", "My MacBook Pro keyboard keys 'E' and 'Space' are repeating or not registering.", "auto", "Keyboard Service Program for butterfly keyboards details.", "medium", "conv_gold_086"),
    ("hardware_repair_issue", "iPhone frame is visibly bent after sitting down with it in my back pocket.", "escalate", "Physical enclosure deformation requiring repair center inspection.", "hard", "conv_gold_087"),
    ("hardware_repair_issue", "Can an unauthorized third-party repair shop replace my iPhone screen without voiding AppleCare?", "auto", "AppleCare third-party modification policy explanation.", "medium", "conv_gold_088"),
    ("hardware_repair_issue", "Lightning port has lint stuck inside and charging cable falls out easily.", "auto", "Safe port cleaning guidelines (non-conductive wooden/plastic pick).", "easy", "conv_gold_089"),
    ("hardware_repair_issue", "Rear camera lens glass cracked, but photos still look normal.", "auto", "Camera glass repair pricing and service appointment link.", "easy", "conv_gold_090"),
    ("hardware_repair_issue", "Power button is jammed and stuck pressed in. I cannot turn on the phone.", "escalate", "Physical button jam requiring hardware repair.", "hard", "conv_gold_091"),
    ("hardware_repair_issue", "How do I check if my device is still covered under AppleCare+ warranty?", "auto", "checkcoverage.apple.com serial number tool link.", "easy", "conv_gold_092"),
    ("hardware_repair_issue", "My iPad screen turned completely purple with vertical lines after being stepped on.", "escalate", "Physical LCD display fracture requiring hardware screen replacement.", "hard", "conv_gold_093"),
    ("hardware_repair_issue", "Vibration Taptic engine makes a loud buzzing grinding noise whenever I get a text.", "escalate", "Mechanical actuator failure requiring internal module replacement.", "hard", "conv_gold_094"),
    ("hardware_repair_issue", "Can I walk into an Apple Store for repair without making an appointment first?", "auto", "Walk-in availability policy vs recommended reservation.", "easy", "conv_gold_095"),
    ("hardware_repair_issue", "What do I need to bring with me to an Apple Store repair appointment?", "auto", "Backup data, disable Find My iPhone, bring photo ID checklist.", "easy", "conv_gold_096"),
    ("hardware_repair_issue", "Screen repair completed yesterday but now True Tone display option is missing.", "escalate", "Post-repair calibration defect requiring revisit to repair center.", "hard", "conv_gold_097"),
    ("hardware_repair_issue", "Does AppleCare+ cover accidental water damage with a service fee?", "auto", "AppleCare+ accidental damage tier pricing details.", "easy", "conv_gold_098"),
    ("hardware_repair_issue", "Sim tray broke off inside the slot and is stuck.", "escalate", "Physical extraction requiring Apple Store technician tools.", "hard", "conv_gold_099"),
    ("hardware_repair_issue", "How long does a mail-in repair take from the day I ship the box?", "auto", "Estimated repair turnaround time (3-5 business days).", "easy", "conv_gold_100"),

    # 6. Cellular, Wi-Fi & Bluetooth Connectivity (20 items)
    ("connectivity_network_issue", "My iPhone suddenly says 'No SIM Card Installed' even though the SIM is inside.", "auto", "Reseat SIM card, restart device, check carrier settings update.", "easy", "conv_gold_101"),
    ("connectivity_network_issue", "Wi-Fi toggle in Control Center is greyed out and cannot be turned on.", "escalate", "Hardware Wi-Fi module failure (greyed out toggle).", "hard", "conv_gold_102"),
    ("connectivity_network_issue", "iPhone connects to home Wi-Fi but says 'No Internet Connection' while other laptops work.", "auto", "Renew DHCP lease, Forget Network, Reset Network Settings.", "easy", "conv_gold_103"),
    ("connectivity_network_issue", "AirPods disconnect from Bluetooth every 3 minutes during phone calls.", "auto", "Reset AirPods in case and Forget Device re-pairing steps.", "easy", "conv_gold_104"),
    ("connectivity_network_issue", "Personal Hotspot is missing from Settings menu on iOS 11.", "auto", "Carrier profile provisioning and Cellular Data APN check.", "medium", "conv_gold_105"),
    ("connectivity_network_issue", "Bluetooth won't pair with my car stereo system after updating phone.", "auto", "Car stereo firmware and Bluetooth pairing profile reset.", "medium", "conv_gold_106"),
    ("connectivity_network_issue", "Phone is stuck on 'Searching...' in status bar and drains battery.", "auto", "Carrier update, SIM test, and iPhone 7 'No Service' program advisory.", "medium", "conv_gold_107"),
    ("connectivity_network_issue", "Can I use 5GHz Wi-Fi network on iPhone 6?", "auto", "Dual-band 802.11a/b/g/n/ac Wi-Fi specs.", "easy", "conv_gold_108"),
    ("connectivity_network_issue", "Cellular data stopped working for all apps except Safari.", "auto", "Check Settings > Cellular individual app toggles.", "easy", "conv_gold_109"),
    ("connectivity_network_issue", "AirDrop is not discovering nearby devices or showing my contact.", "auto", "AirDrop Receiving setting (Everyone vs Contacts Only) & Wi-Fi/BT check.", "easy", "conv_gold_110"),
    ("connectivity_network_issue", "Why does Wi-Fi automatically turn back on at 5 AM on iOS 11?", "auto", "iOS 11 Control Center Wi-Fi disconnect vs turn off behavior explanation.", "easy", "conv_gold_111"),
    ("connectivity_network_issue", "Calls fail immediately as soon as I dial any number on T-Mobile.", "auto", "Airplane mode toggle, Reset Network Settings, Carrier contact.", "medium", "conv_gold_112"),
    ("connectivity_network_issue", "Bluetooth accessory says 'Pairing took too long' error.", "auto", "Accessory discovery mode reset and restart device.", "easy", "conv_gold_113"),
    ("connectivity_network_issue", "Does iPhone 8 support Dual SIM cards?", "auto", "Hardware specification answer for iPhone 8 (single Nano-SIM).", "easy", "conv_gold_114"),
    ("connectivity_network_issue", "Wi-Fi speeds on my iPhone are 2 Mbps while my Mac on same network gets 100 Mbps.", "auto", "DNS settings check, VPN disable test, network reset.", "medium", "conv_gold_115"),
    ("connectivity_network_issue", "LTE icon appears but apps say network timeout.", "auto", "Toggle LTE to 3G/Data Roaming check and carrier provisioning.", "medium", "conv_gold_116"),
    ("connectivity_network_issue", "Apple Watch disconnects from iPhone whenever I walk into another room.", "auto", "Bluetooth range expectations and Wi-Fi handoff guidance.", "easy", "conv_gold_117"),
    ("connectivity_network_issue", "My phone won't connect to enterprise WPA2 802.1X Wi-Fi at university.", "auto", "Certificate trust configuration and network profile installation.", "medium", "conv_gold_118"),
    ("connectivity_network_issue", "GPS location is off by 50 miles in Maps when using cellular data.", "auto", "Location Services calibration and Date & Time automatic setting.", "medium", "conv_gold_119"),
    ("connectivity_network_issue", "Cellular baseband modem chip died completely after liquid exposure.", "escalate", "Baseband hardware failure requiring device exchange.", "hard", "conv_gold_120"),

    # 7. App Store & Application Crashes (20 items)
    ("app_store_app_issue", "App Store keeps giving error 'Cannot connect to App Store' when I open it.", "auto", "Check System Status page, Date & Time automatic, sign out of Media & Purchases.", "easy", "conv_gold_121"),
    ("app_store_app_issue", "WhatsApp closes immediately every time I tap the app icon.", "auto", "Update app, offload app in storage, or backup and reinstall.", "easy", "conv_gold_122"),
    ("app_store_app_issue", "Apps are stuck on 'Waiting...' with dark icon and won't resume download.", "auto", "Pause and resume download, check available storage, restart device.", "easy", "conv_gold_123"),
    ("app_store_app_issue", "How do I change my App Store country/region from UK to US?", "auto", "Cancel active subscriptions, spend balance, update region in settings.", "medium", "conv_gold_124"),
    ("app_store_app_issue", "App Store asks for Touch ID / password for every single free app download.", "auto", "Settings > Face ID & Passcode / Media & Purchases password settings.", "easy", "conv_gold_125"),
    ("app_store_app_issue", "Instagram says 'App requires iOS 12 or later' but my iPhone 5 cannot update to iOS 12.", "auto", "Download last compatible version from Purchased history guide.", "medium", "conv_gold_126"),
    ("app_store_app_issue", "Why are all 32-bit apps showing 'App Needs to Be Updated' popup on iOS 11?", "auto", "iOS 11 64-bit architecture requirement explanation.", "easy", "conv_gold_127"),
    ("app_store_app_issue", "App Store search tab is completely blank white screen.", "auto", "Force quit App Store, clear App Store cache, restart phone.", "medium", "conv_gold_128"),
    ("app_store_app_issue", "How do I delete an app when the 'X' button doesn't appear on home screen?", "auto", "Check Screen Time Restrictions for 'Deleting Apps' toggle.", "medium", "conv_gold_129"),
    ("app_store_app_issue", "Can I set automatic app updates to only happen when connected to Wi-Fi?", "auto", "Settings > App Store > Cellular Data automatic downloads toggle.", "easy", "conv_gold_130"),
    ("app_store_app_issue", "Downloaded a banking app that steals passwords, remove it from App Store immediately!", "escalate", "Malware / fraudulent app security report requiring App Store review team.", "high_risk", "conv_gold_131"),
    ("app_store_app_issue", "App Store says 'Your account is not valid for use in the U.S. store.'", "auto", "Switch store country prompt or match billing address.", "medium", "conv_gold_132"),
    ("app_store_app_issue", "How do I offload unused apps to save space without losing my saved documents?", "auto", "Settings > General > iPhone Storage 'Offload Unused Apps' guide.", "easy", "conv_gold_133"),
    ("app_store_app_issue", "Facebook app freezes whenever I try to upload a photo from camera roll.", "auto", "Settings > Privacy > Photos permission check for Facebook.", "easy", "conv_gold_134"),
    ("app_store_app_issue", "Why is my purchased tab in App Store empty even though I have bought 50 apps?", "auto", "Check if apps were hidden or if signed into correct Apple ID.", "medium", "conv_gold_135"),
    ("app_store_app_issue", "How do I leave a review for an app on the App Store?", "auto", "Navigate to app product page Ratings & Reviews section.", "easy", "conv_gold_136"),
    ("app_store_app_issue", "App Store repeatedly prompts for password loop every 10 seconds.", "auto", "Sign out of Media & Purchases, reboot, sign back in.", "medium", "conv_gold_137"),
    ("app_store_app_issue", "Can I share in-app purchases with family members via Family Sharing?", "auto", "Family Sharing in-app purchase eligibility policy.", "easy", "conv_gold_138"),
    ("app_store_app_issue", "Third party app won't open and crashes entire phone into black screen with spinning gear.", "escalate", "Severe OS springboard crash caused by faulty app.", "hard", "conv_gold_139"),
    ("app_store_app_issue", "How do I stop App Store rating popups inside apps?", "auto", "Settings > App Store > In-App Ratings & Reviews toggle.", "easy", "conv_gold_140"),

    # 8. iCloud Storage, Photos & Syncing (20 items)
    ("icloud_backup_issue", "iCloud Backup failed: 'The last backup could not be completed due to poor network.'", "auto", "Verify Wi-Fi stability, power connection, and sufficient iCloud storage.", "easy", "conv_gold_141"),
    ("icloud_backup_issue", "My photos from today are not appearing on my Mac or iPad iCloud Photo Library.", "auto", "Check Low Power Mode, Wi-Fi connection, and iCloud Photos toggle.", "easy", "conv_gold_142"),
    ("icloud_backup_issue", "I upgraded to 200GB iCloud plan but my iPhone still says iCloud storage is full.", "auto", "Sign out and sign back in to refresh storage quota badge.", "medium", "conv_gold_143"),
    ("icloud_backup_issue", "How do I download all my original full-resolution photos from iCloud to my PC?", "auto", "iCloud for Windows / icloud.com photo download guide.", "easy", "conv_gold_144"),
    ("icloud_backup_issue", "I deleted photos from my iPhone to make space and they disappeared from iCloud too!", "auto", "Explain iCloud sync vs archival backup & check Recently Deleted folder.", "medium", "conv_gold_145"),
    ("icloud_backup_issue", "iCloud Drive documents are stuck on 'Uploading 1 item' for 3 weeks.", "auto", "Toggle iCloud Drive off/on and inspect file size limits.", "medium", "conv_gold_146"),
    ("icloud_backup_issue", "My iPhone backup size is 12GB but I only have 5GB free iCloud storage. What do I do?", "auto", "Manage Backup size in settings or upgrade storage tier.", "easy", "conv_gold_147"),
    ("icloud_backup_issue", "All my notes disappeared after turning off iCloud Notes toggle.", "auto", "Turn iCloud Notes toggle back on to re-sync from server.", "easy", "conv_gold_148"),
    ("icloud_backup_issue", "Can I share my 200GB iCloud storage plan with my family members?", "auto", "Family Sharing iCloud storage allocation instructions.", "easy", "conv_gold_149"),
    ("icloud_backup_issue", "How do I recover deleted contacts from iCloud.com?", "auto", "icloud.com Account Settings > Restore Contacts guide.", "easy", "conv_gold_150"),
    ("icloud_backup_issue", "iCloud backup restore stopped at 99% and has been stuck there for 24 hours.", "auto", "Restart phone or check app download queue completion.", "medium", "conv_gold_151"),
    ("icloud_backup_issue", "Why does 'System' and 'Other' take up 30GB of space in iCloud backup?", "auto", "Detailed breakdown of diagnostic data and cache in backup.", "medium", "conv_gold_152"),
    ("icloud_backup_issue", "I lost all irreplaceable photos of my deceased mother after an iCloud sync error!", "escalate", "Severe data loss distress requiring senior data recovery specialist.", "high_risk", "conv_gold_153"),
    ("icloud_backup_issue", "How do I turn off iCloud Photo Library without deleting photos from my phone?", "auto", "'Download and Keep Originals' selection before turning off guide.", "medium", "conv_gold_154"),
    ("icloud_backup_issue", "Messages in iCloud is taking 10GB of storage. How do I delete old attachments?", "auto", "Settings > General > iPhone Storage > Messages review large attachments.", "easy", "conv_gold_155"),
    ("icloud_backup_issue", "Can I back up my iPhone to an external USB hard drive directly from the phone?", "auto", "Clarify iTunes/Finder Mac/PC requirement for external drive backup.", "easy", "conv_gold_156"),
    ("icloud_backup_issue", "iCloud says 'Verification Failed. An error occurred connecting to iCloud.'", "auto", "Check Apple System Status page and restart network connection.", "easy", "conv_gold_157"),
    ("icloud_backup_issue", "How do I downgrade my iCloud storage plan back to the free 5GB tier?", "auto", "Manage Storage > Change Storage Plan > Downgrade Options guide.", "easy", "conv_gold_158"),
    ("icloud_backup_issue", "Does WhatsApp backup to iCloud count against my free 5GB iCloud storage?", "auto", "Confirm WhatsApp backup consumes iCloud Drive quota.", "easy", "conv_gold_159"),
    ("icloud_backup_issue", "My shared iCloud photo album is not sending invite notifications to my friends.", "auto", "Verify recipient Apple ID email and Shared Albums settings.", "medium", "conv_gold_160"),

    # 9. Audio, Microphone & Media Playback (20 items)
    ("audio_media_issue", "People cannot hear me when I speak on phone calls unless I switch to speakerphone.", "auto", "Microphone grill cleaning and voice memo hardware test.", "easy", "conv_gold_161"),
    ("audio_media_issue", "Top receiver ear speaker is extremely muffled during calls on iPhone 7.", "auto", "Receiver mesh cleaning and sound balance settings check.", "easy", "conv_gold_162"),
    ("audio_media_issue", "Songs on Apple Music keep pausing randomly after 15 seconds of playback.", "auto", "Download song offline test, sign out/in of Apple Music, network check.", "easy", "conv_gold_163"),
    ("audio_media_issue", "One side of my AirPods (left earbud) has zero sound and won't charge in case.", "auto", "Clean charging contacts, reset AirPods case button for 15 seconds.", "easy", "conv_gold_164"),
    ("audio_media_issue", "My phone is stuck in 'Headphones' mode even though nothing is plugged in.", "auto", "Inspect lightning port for debris, plug/unplug headphones, reboot.", "medium", "conv_gold_165"),
    ("audio_media_issue", "Audio is distorted and crackles loudly whenever volume is set above 50%.", "escalate", "Blown physical speaker cone requiring hardware replacement.", "hard", "conv_gold_166"),
    ("audio_media_issue", "Why does volume automatically drop when I open camera or game?", "auto", "Change with Buttons toggle and attention awareness settings.", "easy", "conv_gold_167"),
    ("audio_media_issue", "Siri cannot hear my voice when I say 'Hey Siri'.", "auto", "Retrain Hey Siri voice model in Settings > Siri & Search.", "easy", "conv_gold_168"),
    ("audio_media_issue", "How do I EQ bass boost on Apple Music on iPhone?", "auto", "Settings > Music > EQ > Bass Booster guide.", "easy", "conv_gold_169"),
    ("audio_media_issue", "My AirPods volume is very quiet even when iPhone volume slider is at 100%.", "auto", "Audio Balance in Accessibility and Volume Limit settings check.", "easy", "conv_gold_170"),
    ("audio_media_issue", "Microphone makes high-pitched deafening screeching feedback that hurt my eardrum!", "escalate", "Audio feedback injury risk requiring immediate hardware diagnostic.", "high_risk", "conv_gold_171"),
    ("audio_media_issue", "Bluetooth audio is out of sync with video when watching Netflix or YouTube.", "auto", "App cache refresh, reconnect Bluetooth device, low latency settings.", "medium", "conv_gold_172"),
    ("audio_media_issue", "Why does music stop playing whenever I receive a text notification?", "auto", "Sound & Haptics notification ducking behavior explanation.", "easy", "conv_gold_173"),
    ("audio_media_issue", "Podcast app won't play downloaded episodes offline on airplane mode.", "auto", "Podcast continuous playback and cellular data settings check.", "medium", "conv_gold_174"),
    ("audio_media_issue", "Can I connect two pairs of AirPods to one iPhone at the same time?", "auto", "Audio Sharing feature compatibility requirements.", "easy", "conv_gold_175"),
    ("audio_media_issue", "Bottom right speaker has no sound coming out on my iPhone.", "auto", "Clarify bottom right is speaker, bottom left is microphone.", "easy", "conv_gold_176"),
    ("audio_media_issue", "My ringtone is silent even though the mute switch is off and volume is up.", "auto", "Check Do Not Disturb / Focus mode status.", "easy", "conv_gold_177"),
    ("audio_media_issue", "Voice Memos app says 'Audio recording failed' when tapping red record button.", "auto", "Microphone privacy permission and storage space check.", "medium", "conv_gold_178"),
    ("audio_media_issue", "Lightning to 3.5mm headphone adapter is not recognized: 'Accessory not supported'.", "auto", "MFi certification explanation and port cleaning.", "easy", "conv_gold_179"),
    ("audio_media_issue", "Audio keeps switching back and forth between speaker and earpiece during calls.", "auto", "Proximity sensor check and case obstruction test.", "medium", "conv_gold_180"),

    # 10. General Inquiry, Feature Guidance & Store Advice (20 items)
    ("general_inquiry_advice", "Do you offer customer support on Twitter in Spanish or Portuguese?", "auto", "Direct user to international Twitter handles or localized support portals.", "easy", "conv_gold_181"),
    ("general_inquiry_advice", "How do I take a screenshot on the new iPhone X without a home button?", "auto", "Press Side button + Volume Up button simultaneously.", "easy", "conv_gold_182"),
    ("general_inquiry_advice", "Can I trade in my old iPhone 6 at the Apple Store towards an iPhone 8?", "auto", "Apple Trade In program valuation and store trade-in guide.", "easy", "conv_gold_183"),
    ("general_inquiry_advice", "What are the retail store opening hours for the Regent Street Apple Store in London?", "auto", "Apple Store locator and official hours portal link.", "easy", "conv_gold_184"),
    ("general_inquiry_advice", "Is iPhone 8 waterproof if I take it swimming in the ocean?", "auto", "IP67 water resistance rating explanation (freshwater vs saltwater).", "easy", "conv_gold_185"),
    ("general_inquiry_advice", "How do I enable Dark Mode or Invert Colors on iOS 11?", "auto", "Settings > General > Accessibility > Display Accommodations > Smart Invert.", "easy", "conv_gold_186"),
    ("general_inquiry_advice", "I am writing a research thesis on Apple's corporate supply chain. Who can I interview?", "auto", "Direct to Apple PR / media relations portal.", "medium", "conv_gold_187"),
    ("general_inquiry_advice", "How do I record my iPhone screen with the new iOS 11 Control Center tool?", "auto", "Settings > Control Center > Customize Controls > Screen Recording guide.", "easy", "conv_gold_188"),
    ("general_inquiry_advice", "Can Apple pencil work on iPhone or only on iPad Pro?", "auto", "Apple Pencil hardware compatibility specifications (iPad only).", "easy", "conv_gold_189"),
    ("general_inquiry_advice", "I have an idea for an unreleased Apple product patent, sign my NDA immediately!", "escalate", "Unsolicited product idea / legal NDA submission requiring legal intake.", "high_risk", "conv_gold_190"),
    ("general_inquiry_advice", "How do I turn on True Tone display on iPad Pro?", "auto", "Settings > Display & Brightness > True Tone toggle.", "easy", "conv_gold_191"),
    ("general_inquiry_advice", "Where can I recycle my old broken iPod and laptop cables responsibly?", "auto", "Apple recycling program and free in-store drop-off details.", "easy", "conv_gold_192"),
    ("general_inquiry_advice", "How do I customize Control Center shortcuts on iOS 11?", "auto", "Settings > Control Center > Customize Controls guide.", "easy", "conv_gold_193"),
    ("general_inquiry_advice", "Does the iPhone 8 come with fast charger in the box?", "auto", "Package contents specs (standard 5W adapter included).", "easy", "conv_gold_194"),
    ("general_inquiry_advice", "How do I use Reachability on iPhone without home button?", "auto", "Swipe down on bottom edge of screen guide.", "easy", "conv_gold_195"),
    ("general_inquiry_advice", "Can I buy AppleCare+ after 30 days of purchasing my iPhone?", "auto", "AppleCare+ 60-day purchase window policy with diagnostic check.", "medium", "conv_gold_196"),
    ("general_inquiry_advice", "Are student discounts available on iPad Pro purchases at the education store?", "auto", "Apple Education Pricing eligibility and portal link.", "easy", "conv_gold_197"),
    ("general_inquiry_advice", "What is the return policy window if I bought an iPhone at the Apple Store?", "auto", "Standard 14-day return/exchange policy explanation.", "easy", "conv_gold_198"),
    ("general_inquiry_advice", "Can I use Apple Pay in Germany or which countries are currently supported?", "auto", "Apple Pay country and region availability list portal.", "easy", "conv_gold_199"),
    ("general_inquiry_advice", "How do I scan QR codes using the camera app on iOS 11?", "auto", "Point camera at QR code and tap notification banner guide.", "easy", "conv_gold_200"),
]

def main():
    records = []
    for i, item in enumerate(GOLDEN_ITEMS, start=1):
        intent, msg, action, reason, diff, conv_id = item
        records.append({
            "id": f"gold_{i:03d}",
            "customer_message": msg,
            "intent": intent,
            "expected_action": action,
            "expected_reason": reason,
            "difficulty": diff,
            "source_conversation_id": conv_id
        })

    df = pd.DataFrame(records)
    out_csv = ROOT_DIR / "evaluation" / "golden_set.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    logger.info(f"Successfully generated Golden Evaluation Set with exactly {len(df)} examples at {out_csv}.")
    print("\nGolden Set Distribution by Intent:")
    print(df["intent"].value_counts())
    print("\nGolden Set Distribution by Expected Action:")
    print(df["expected_action"].value_counts())
    print("\nGolden Set Distribution by Difficulty:")
    print(df["difficulty"].value_counts())

if __name__ == "__main__":
    main()
