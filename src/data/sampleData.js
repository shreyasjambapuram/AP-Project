export const SAMPLE_MEMBERS = [
  { id: '1', name: 'Alex Chen', grade: 11, events: ['astronomy', 'dynamic-planet', 'remote-sensing'], hours: 42, email: 'alex.c@school.edu' },
  { id: '2', name: 'Priya Sharma', grade: 10, events: ['anatomy', 'disease-detectives', 'forensics'], hours: 38, email: 'priya.s@school.edu' },
  { id: '3', name: 'Marcus Johnson', grade: 12, events: ['chemistry-lab', 'experimental-design'], hours: 55, email: 'marcus.j@school.edu' },
  { id: '4', name: 'Sophie Williams', grade: 11, events: ['ecology', 'dynamic-planet', 'fossils'], hours: 31, email: 'sophie.w@school.edu' },
  { id: '5', name: 'David Kim', grade: 10, events: ['codebusters', 'robot-tour'], hours: 47, email: 'david.k@school.edu' },
  { id: '6', name: 'Emma Rodriguez', grade: 9, events: ['write-it-do-it', 'fossils'], hours: 12, email: 'emma.r@school.edu' },
  { id: '7', name: 'James Liu', grade: 12, events: ['optics', 'wind-power', 'towers'], hours: 60, email: 'james.l@school.edu' },
  { id: '8', name: 'Aisha Patel', grade: 11, events: ['microbe-mission', 'anatomy'], hours: 35, email: 'aisha.p@school.edu' },
  { id: '9', name: 'Ryan O\'Brien', grade: 10, events: ['helicopter', 'towers', 'wind-power'], hours: 28, email: 'ryan.o@school.edu' },
  { id: '10', name: 'Lily Zhang', grade: 9, events: ['ecology', 'write-it-do-it'], hours: 15, email: 'lily.z@school.edu' },
  { id: '11', name: 'Carlos Martinez', grade: 11, events: ['remote-sensing', 'astronomy'], hours: 44, email: 'carlos.m@school.edu' },
  { id: '12', name: 'Nina Kowalski', grade: 12, events: ['experimental-design', 'chemistry-lab', 'forensics'], hours: 52, email: 'nina.k@school.edu' },
];

export const SAMPLE_DEADLINES = [
  { id: '1', title: 'Officer Applications Due', date: '2026-04-20', category: 'Admin', notes: 'Submit via Google Form for 2026-27' },
  { id: '2', title: 'NestSO Tests Due', date: '2026-03-09', category: 'Tournament', notes: 'Satellite testing closes' },
  { id: '3', title: 'Mid-Year Feedback Form', date: '2025-12-20', category: 'Admin', notes: 'Anonymous feedback for officers' },
  { id: '4', title: 'Districts Registration Due', date: '2025-12-08', category: 'Tournament', notes: 'Sign-up for CyWoods' },
  { id: '5', title: 'Seven Lakes Release Forms & $5', date: '2025-12-05', category: 'Admin', notes: 'Mandatory for competition' },
  { id: '6', title: 'Monta Vista Tests Due', date: '2025-11-20', category: 'Tournament', notes: 'All satellite events must be completed' },
  { id: '7', title: 'Rickards Tests Due', date: '2025-11-08', category: 'Tournament', notes: 'Submit before 11PM' },
  { id: '8', title: 'UT Austin Release Forms', date: '2025-10-18', category: 'Admin', notes: 'Must return signed forms to Mrs. Turner' },
];

export const SAMPLE_STUDY_HOURS = [
  { id: '1', memberId: '1', memberName: 'Alex Chen', date: '2026-04-28', duration: 2.5, event: 'astronomy', notes: 'HR diagram practice' },
  { id: '2', memberId: '1', memberName: 'Alex Chen', date: '2026-04-25', duration: 1.5, event: 'dynamic-planet', notes: 'Glacier formations review' },
  { id: '3', memberId: '2', memberName: 'Priya Sharma', date: '2026-04-28', duration: 3, event: 'anatomy', notes: 'Cardiovascular system' },
  { id: '4', memberId: '3', memberName: 'Marcus Johnson', date: '2026-04-27', duration: 2, event: 'chemistry-lab', notes: 'Titration practice' },
  { id: '5', memberId: '5', memberName: 'David Kim', date: '2026-04-28', duration: 4, event: 'robot-tour', notes: 'Navigation calibration' },
  { id: '6', memberId: '7', memberName: 'James Liu', date: '2026-04-26', duration: 3, event: 'optics', notes: 'Lens equations' },
  { id: '7', memberId: '4', memberName: 'Sophie Williams', date: '2026-04-29', duration: 1.5, event: 'ecology', notes: 'Biome identification' },
  { id: '8', memberId: '6', memberName: 'Emma Rodriguez', date: '2026-04-29', duration: 1, event: 'fossils', notes: 'Fossil ID practice' },
  { id: '9', memberId: '11', memberName: 'Carlos Martinez', date: '2026-04-28', duration: 2, event: 'remote-sensing', notes: 'False-color analysis' },
  { id: '10', memberId: '12', memberName: 'Nina Kowalski', date: '2026-04-27', duration: 2.5, event: 'forensics', notes: 'Blood spatter analysis' },
];

export const SAMPLE_MEETING_LINKS = [
  { id: 'm1', date: '2026-04-06', title: 'Weekly Team Meeting - Week 1', slidesUrl: '#', minutesUrl: '#' },
  { id: 'm2', date: '2026-04-13', title: 'Weekly Team Meeting - Week 2', slidesUrl: '#', minutesUrl: '#' },
  { id: 'm3', date: '2026-04-20', title: 'Weekly Team Meeting - Week 3', slidesUrl: '#', minutesUrl: '#' },
  { id: 'm4', date: '2026-04-27', title: 'Weekly Team Meeting - Week 4', slidesUrl: '#', minutesUrl: '#' },
];

export const SAMPLE_SIGNUP_LINKS = [
  { id: '1', title: 'Officer Applications 2026', category: 'Admin', url: 'https://forms.gle/JPJr2QW1npX76Ny58', date: '2026-04-20', status: 'Closed' },
  { id: '2', title: 'NestSO Satellite', category: 'Tournament', url: 'https://docs.google.com/forms/d/e/1FAIpQLSfzPngRWdMzPJ-qYU-ykzkhhJlUGO_5Tuai5dlc06sD5HJCyA/viewform', date: '2026-03-09', status: 'Closed' },
  { id: '3', title: 'Districts (CyWoods HS)', category: 'Tournament', url: 'https://docs.google.com/forms/d/1ylvVoMiOQj5ZbpT5SJVr_H7N2_VarQQJKPK6A7r90QY/edit', date: '2026-01-17', status: 'Closed' },
  { id: '4', title: 'Seven Lakes Invitational', category: 'Tournament', url: 'https://docs.google.com/forms/d/e/1FAIpQLSetntmq45Y3GNu1_6G2IlOns7OR2S1B9vOFcgby9oIuaf9kug/viewform', date: '2025-12-06', status: 'Closed' },
  { id: '5', title: 'Monta Vista Satellite', category: 'Tournament', url: 'https://docs.google.com/forms/d/e/1FAIpQLSd_5BpfXJdIIEoP8awYWSbCMrAyCP3m_dHfqAz0TVJze1-4Tw/viewform', date: '2025-11-14', status: 'Closed' },
  { id: '6', title: 'Rickards Satellite', category: 'Tournament', url: 'https://forms.gle/Btb2sG8kF76cNq2y5', date: '2025-11-01', status: 'Closed' },
  { id: '7', title: 'Georgia Scrimmage', category: 'Tournament', url: 'https://m.signupgenius.com/#!/showSignUp/10C094AAEA723A4FEC25-59068720-georgia', date: '2025-10-11', status: 'Closed' },
  { id: '8', title: 'Highlands Virtual Invitational', category: 'Tournament', url: 'https://m.signupgenius.com/#!/showSignUp/10C094AAEA723A4FEC25-59068325-highlands', date: '2025-10-04', status: 'Closed' },
  { id: '9', title: 'Local Table Spirit Night', category: 'Social', url: '#', date: '2025-10-14', status: 'Closed' },
];

export const SAMPLE_INVENTORY = [
  { id: '1', name: 'Dynamic Planet Binder', category: 'Study', event: 'dynamic-planet', quantity: 3, location: 'Cabinet A', condition: 'Good' },
  { id: '2', name: 'Titration Kit', category: 'Lab', event: 'chemistry-lab', quantity: 2, location: 'Lab Storage', condition: 'Good' },
  { id: '3', name: 'Helicopter Kit (Balsa Wood)', category: 'Build', event: 'helicopter', quantity: 5, location: 'Build Room', condition: 'New' },
  { id: '4', name: 'Arduino Starter Pack', category: 'Build', event: 'robot-tour', quantity: 3, location: 'Build Room', condition: 'Good' },
  { id: '5', name: 'Fossil Specimen Set', category: 'Study', event: 'fossils', quantity: 1, location: 'Cabinet B', condition: 'Fair' },
  { id: '6', name: 'Optics Lens Kit', category: 'Lab', event: 'optics', quantity: 2, location: 'Lab Storage', condition: 'Good' },
  { id: '7', name: 'Wind Turbine Motor Set', category: 'Build', event: 'wind-power', quantity: 4, location: 'Build Room', condition: 'Good' },
  { id: '8', name: 'Anatomy Model (Torso)', category: 'Study', event: 'anatomy', quantity: 1, location: 'Cabinet A', condition: 'Good' },
  { id: '9', name: 'AA Batteries (Pack of 24)', category: 'Build', event: 'robot-tour', quantity: 2, location: 'Supply Closet', condition: 'New' },
  { id: '10', name: 'Astro Reference Textbook', category: 'Study', event: 'astronomy', quantity: 2, location: 'Cabinet B', condition: 'Fair' },
];

export const SAMPLE_CHECKOUTS = [
  { id: '1', memberName: 'Alex Chen', itemName: 'Astro Reference Textbook', checkoutDate: '2026-04-20', returnDate: '2026-05-04', status: 'Checked Out', notes: 'For invitational prep' },
  { id: '2', memberName: 'David Kim', itemName: 'Arduino Starter Pack', checkoutDate: '2026-04-15', returnDate: '2026-04-29', status: 'Overdue', notes: 'Robot Tour build' },
  { id: '3', memberName: 'Sophie Williams', itemName: 'Dynamic Planet Binder', checkoutDate: '2026-04-22', returnDate: '2026-05-06', status: 'Checked Out', notes: '' },
];

export const SAMPLE_SUPPLY_REQUESTS = [
  { id: '1', memberName: 'Ryan O\'Brien', item: '9V Batteries', event: 'helicopter', urgency: 'High', status: 'Pending', notes: 'Need for regional build' },
  { id: '2', memberName: 'Emma Rodriguez', item: '3-Ring Binder (2")', event: 'fossils', urgency: 'Medium', status: 'Approved', notes: 'For fossil ID binder' },
  { id: '3', memberName: 'James Liu', item: 'Replacement Lens (50mm convex)', event: 'optics', urgency: 'High', status: 'Fulfilled', notes: 'Old one cracked' },
];

export const SAMPLE_NOTES = [
  { id: '1', title: 'Stellar Evolution Complete Guide', event: 'astronomy', season: '2025-2026', author: 'Carlos Martinez', description: 'Comprehensive notes covering main sequence, giants, white dwarfs, neutron stars, and black holes.', url: '#' },
  { id: '2', title: 'Dynamic Planet - Glaciers Cheat Sheet', event: 'dynamic-planet', season: '2025-2026', author: 'Alex Chen', description: 'Quick-reference sheet for glacier types, formations, and associated landforms.', url: '#' },
  { id: '3', title: 'Anatomy: Nervous System Deep Dive', event: 'anatomy', season: '2024-2025', author: 'Priya Sharma', description: 'Detailed notes on CNS, PNS, neurotransmitters, and reflex arcs.', url: '#' },
  { id: '4', title: 'Remote Sensing Band Combinations', event: 'remote-sensing', season: '2025-2026', author: 'Carlos Martinez', description: 'Guide to Landsat and Sentinel band combinations for vegetation, water, and urban analysis.', url: '#' },
  { id: '5', title: 'Codebusters Cipher Handbook', event: 'codebusters', season: '2024-2025', author: 'David Kim', description: 'Reference for all cipher types including Hill cipher matrices and Baconian cipher.', url: '#' },
];

export const SAMPLE_PRACTICE_TESTS = [
  { id: '1', title: '7 Lakes', difficulty: 'Invitational', year: '2025', url: 'http://drive.google.com/drive/folders/1cZVE8xtgVZrh8nOsyyZMenTw8cvatutz' },
  { id: '2', title: 'Columbia', difficulty: 'Invitational', year: '2025', url: 'https://drive.google.com/drive/folders/1_45wLrlBYvAtD8IqfcMET4_IsO7gu1e7' },
  { id: '3', title: 'CSE', difficulty: 'Invitational', year: '2025', url: 'https://drive.google.com/drive/folders/1jV4llwlWYd8_lxoPWyw2oE7IJxFQMko0' },
  { id: '4', title: 'GGSO', difficulty: 'Invitational', year: '2025', url: 'https://drive.google.com/drive/folders/1twXDpWKFKFqNzL8tZzLlpJo4hhR6NYnP' },
  { id: '5', title: 'GullSO', difficulty: 'Invitational', year: '2025', url: 'https://drive.google.com/drive/folders/1_Eq1W3KQ-o1WKK6Ifhav2fpn4TLJzryw' },
  { id: '6', title: 'H&H', difficulty: 'Invitational', year: '2025', url: 'https://drive.google.com/drive/folders/1a73BgVxk-dZ9Ozx6Jd5Pjnb7iaHLr5Kg' },
  { id: '7', title: 'Kentuckiana', difficulty: 'Invitational', year: '2025', url: 'https://sites.google.com/jefferson.kyschools.us/scioly-ky-in-invite/tests-keys?authuser=0' },
  { id: '8', title: 'MIT', difficulty: 'Invitational', year: '2025', url: 'https://drive.google.com/drive/folders/1f0zivCCd3M8WaeLuE2biguZzn0282gS5' },
  { id: '9', title: 'Monta Vista', difficulty: 'Invitational', year: '2025', url: 'https://drive.google.com/drive/folders/1sqOQPNkJKrCCGT00IyV7sZWeaYqyuzm0' },
  { id: '10', title: 'Northview', difficulty: 'Invitational', year: '2025', url: 'https://drive.google.com/drive/folders/1FU-fEdq3SsbOs2V5aonza-8zysrNwM1H' },
  { id: '11', title: 'Princeton', difficulty: 'Invitational', year: '2025', url: '#' },
  { id: '12', title: 'Purdue', difficulty: 'Invitational', year: '2025', url: '#' },
  { id: '13', title: 'SOAPS', difficulty: 'Invitational', year: '2025', url: '#' },
  { id: '14', title: 'UChicago', difficulty: 'Invitational', year: '2025', url: '#' },
  { id: '15', title: 'UGA', difficulty: 'Invitational', year: '2025', url: '#' },
  { id: '16', title: 'UMSO', difficulty: 'Invitational', year: '2025', url: '#' },
  { id: '17', title: 'USC', difficulty: 'Invitational', year: '2025', url: '#' },
  { id: '18', title: 'UT Austin', difficulty: 'Invitational', year: '2025', url: '#' },
  { id: '19', title: 'VTech', difficulty: 'Invitational', year: '2025', url: '#' },
  { id: '20', title: 'WashU', difficulty: 'Invitational', year: '2025', url: '#' },
  { id: '21', title: 'Yale', difficulty: 'Invitational', year: '2025', url: '#' },
];



export const ELIGIBILITY_THRESHOLD = 20; // minimum hours for tournament eligibility
