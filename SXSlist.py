#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 10:41:46 2019

@author: drizl
"""

DEFAULT_NOSPIN_SXS_LIST = ('0071', '0169', '0168', '0294', '0167', 
                        '0295', '0296', '0166', '0297', '0298',
                        '0299', '0300', '0301', '0302', '0303')
DEFAULT_LOWSPIN_SXS_LIST = ('0222', '0038', '0041', '0004', '0005', '0005', 
                        '0014', '0148', '0170', '0171', '0203', '0205',
                        '0240', '0241', '0242')
DEFAULT_HIGHSPIN_SXS_LIST = ('0025', '0151', '0154', '0155', '0157', '0158',
                        '0159', '0160', '0177', '0178', '0202', '0231', '0280')
DEFAULT_NOSPIN_ECC_SXS_LIST = [str(x) for x in range(1355, 1375)]
DEFAULT_SPIN_ECC_SXS_LIST = ['%04d'%x for x in range(321, 325)]
DEFAULT_ECC_SXS_LIST2 = [str(x) for x in range(2266, 2313)]
DEFAULT_NOSPIN_SXS_LIST2 = ('0055', '0056', '0063', '0066', '0070', '0071', 
                            '0166', '0167', '0168', '0169', '0182', '0183',
                            '0184', '0298', '0301')
DEFAULT_NOSPIN_SXS_LIST3 = ("0180", "0093", "0184",
                        "0259", "0183", "0294",
                        "0182", "0295", "0107",
                        "0296", "0166", "0297",
                        "0298", "0299", "0063",
                        "0186", "0300", "0301",
                        "0302", "0303", "0185")

DEFAULT_NOSPIN_Q1_LIST = ('0066', '0070', '0071', '0180')

DEFAULT_NOSPIN_Q2_LIST = ('0169', '0184')

DEFAULT_NOSPIN_Q3_LIST = ('0168', '0183')

DEFAULT_NOSPIN_Q4_LIST = ('0167', '0182')

DEFAULT_NOSPIN_Q5_LIST = ('0055', '0056')

DEFAULT_NOSPIN_HIGHQ_LIST = ('0166', '0298', '0063', '0301', '0303')

DEFAULT_LOWSPIN_Q1_LIST = ('0222', '0004', '0005', '0218', '0304')

DEFAULT_HIGHSPIN_Q1_LIST = ('0210', '0216', '0227', '0231', '0152', \
                            '0160', '0209', '0211', '0215', '0217', \
                            '0228', '0231')

DEFAULT_LOWSPIN_Q2_LIST = ('0238',  '0240',  '0241', '0242', '0249', '0250', '0251', '0253')

DEFAULT_HIGHSPIN_Q2_LIST = ('0162', '0234', '0235', '0236', '0237', '0254', '0255', '0256', '0257', '0258')

DEFAULT_LOWSPIN_Q3_LIST = ('0031', '0036', '0041', '0271', '0272', '0281', '0282', '0283', '0286', '0045', '0046', '0047')

DEFAULT_HIGHSPIN_Q3_LIST = ('0287', '0288', '0289', '0290', '0291')

DEFAULT_SPIN_Q5_LIST = ('0060', '0061', '0108', '0109', '0110', '0111')

DEFAULT_SPIN_Q8_LIST = ('0064', '0065', '0114', '1419', '1420', '1423', '1426', '1430' '1431', '1433', '1441')

DEFAULT_SPIN_LIST = list(DEFAULT_LOWSPIN_Q1_LIST) + list(DEFAULT_HIGHSPIN_Q1_LIST) + \
    list(DEFAULT_LOWSPIN_Q2_LIST) + list(DEFAULT_HIGHSPIN_Q2_LIST) + \
        list(DEFAULT_LOWSPIN_Q3_LIST) + list(DEFAULT_HIGHSPIN_Q3_LIST) + \
            list(DEFAULT_SPIN_Q5_LIST) + list(DEFAULT_SPIN_Q8_LIST)

DEFAULT_SPIN_LIST2 = [str(x) for x in range(1419, 1510)]
DEFAULT_SPIN_LIST3 = [str(x) for x in range(2083, 2163)]
DEFAULT_HUGESPIN = ('0153', '0154', '0155', '0157', '0158', '0159', '0160', '0172', \
    '0176', '0177', '0178', '0211', '0212', '0213', '0230', '0233', '0234', '0257', \
    '0258', '0260', '0293', '0306')
DEFAULT_SXS_LISTV2_NOECC_ALL = ("0001", "0070", "0004", "0005", "0007", "0008", "0009", \
         "0012", "0013", "0014", "0016", "0019", "0025", "0030", \
         "0031", "0036", "0038", "0039", "0040", "0041", "0045", \
         "0046", "0047", "0054", "0055", "0056", "0060", "0061", \
         "0063", "0064", "0065", "0066", "0083", "0084", "0085", \
         "0086", "0087", "0089", "0090", "0091", "0093", "0100", \
         "0101", "0105", "0106", "0107", "0108", "0109", "0110", \
         "0111", "0112", "0113", "0114", "0148", "0149", "0150", \
         "0151", "0152", "0153", "0154", "0155", "0157", "0158", \
         "0159", "0160", "0162", "0166", "0167", "0168", "0169", \
         "0170", "0171", "0172", "0174", "0175", "0176", "0177", \
         "0178", "0180", "0181", "0182", "0183", "0184", "0185", \
         "0186", "0187", "0188", "0189", "0190", "0191", "0192", \
         "0193", "0194", "0195", "0196", "0197", "0198", "0199", \
         "0200", "0201", "0202", "0203", "0204", "0205", "0206", \
         "0207", "0208", "0209", "0210", "0211", "0212", "0213", \
         "0214", "0215", "0216", "0217", "0218", "0219", "0220", \
         "0221", "0222", "0223", "0224", "0225", "0226", "0227", \
         "0228", "0229", "0230", "0231", "0232", "0233", "0234", \
         "0235", "0236", "0237", "0238", "0239", "0240", "0241", \
         "0242", "0243", "0244", "0245", "0246", "0247", "0248", \
         "0249", "0250", "0251", "0252", "0253", "0254", "0255", \
         "0256", "0257", "0258", "0259", "0260", "0261", "0262", \
         "0263", "0264", "0265", "0266", "0267", "0268", "0269", \
         "0270", "0271", "0272", "0273", "0274", "0275", "0276", \
         "0277", "0278", "0279", "0280", "0281", "0282", "0283", \
         "0284", "0285", "0286", "0287", "0288", "0289", "0290", \
         "0291", "0292", "0293", "0294", "0295", "0296", "0297", \
         "0298", "0299", "0300", "0301", "0302", "0303", "0304", \
         "0305", "0306", "0307", "1419", "1420", "1421", "1422", \
         "1423", "1424", "1425", "1426", "1427", "1428", "1429", \
         "1430", "1431", "1432", "1433", "1434", "1435", "1436", \
         "1437", "1438", "1439", "1440", "1441", "1442", "1443", \
         "1444", "1445", "1446", "1447", "1448", "1449", "1450", \
         "1451", "1452", "1453", "1454", "1455", "1456", "1457", \
         "1458", "1459", "1460", "1461", "1462", "1463", "1464", \
         "1465", "1466", "1467", "1468", "1469", "1470", "1471", \
         "1472", "1473", "1474", "1475", "1476", "1477", "1478", \
         "1479", "1480", "1481", "1482", "1483", "1484", "1485", \
         "1486", "1487", "1488", "1489", "1490", "1491", "1492", \
         "1493", "1494", "1495", "1496", "1497", "1498", "1499", "1500")

DEFAULT_SXS_PREC_LIST = ('0003', '0006', '0010', '0011', '0015', '0017', \
                        '0018', '0020', '0021', '0022', '0023', '0024', \
                        '0026', '0027', '0028', '0029', '0032', '0033', \
                        '0034', '0035', '0037', '0042', '0043', '0044', \
                        '0048', '0049', '0050', '0051', '0052', '0053', \
                        '0057', '0058', '0059', '0062', '0075', '0076', \
                        '0077', '0078', '0079', '0080', '0081', '0082', \
                        '0088', '0092', '0094', '0095', '0096', '0097', \
                        '0098', '0099', '0102', '0103', '0104', '0115', \
                        '0116', '0117', '0118', '0119', '0120', '0121', \
                        '0122', '0123', '0124', '0125', '0126', '0127', \
                        '0128', '0129', '0130', '0131', '0132', '0133', \
                        '0134', '0135', '0136', '0137', '0138', '0139', \
                        '0140', '0141', '0142', '0143', '0144', '0145', \
                        '0146', '0147', '0156', '0161', '0163', '0164', \
                        '0165', '0173', '0179', '0308')

DEFAULT_SXS_PREC_LIST2 = ('0316', '0336', '0337', '0338', '0339', '0340', '0341', '0342', \
                        '0343', '0344', '0345', '0346', '0347', '0348', '0349', '0350', \
                        '0351', '0352', '0353', '0356', '0357', '0358', '0359', '0360', \
                        '0362', '0363', '0364', '0365', '0367', '0368', '0373', '0374', \
                        '0378', '0379', '0380', '0381', '0383', '0384', '0390', '0391', \
                        '0393', '0395', '0396', '0400', '0401', '0403', '0405', '0406', \
                        '0408', '0411', '0413', '0416', '0417', '0419', '0420', '0421', \
                        '0422', '0424', '0425', '0426', '0427', '0428', '0429', '0430', \
                        '0431', '0432', '0433', '0434', '0439', '0442', '0443', '0444', \
                        '0445', '0446', '0449', '0450', '0452', '0453', '0455', '0456', \
                        '0457', '0458', '0460', '0463', '0467', '0468', '0469', '0470', \
                        '0471', '0472', '0474', '0476', '0477', '0478', '0479', '0480', \
                        '0481', '0482', '0483', '0484', '0485', '0487', '0489', '0490', \
                        '0491', '0492', '0493', '0494', '0495', '0496', '0497', '0498', \
                        '0499', '0500', '0502', '0504', '0505', '0506', '0508', '0509', \
                        '0510', '0511', '0514', '0515', '0516', '0517', '0518', '0519', \
                        '0520', '0521', '0522', '0523', '0524', '0526', '0527', '0528', \
                        '0529', '0530', '0531', '0532', '0533', '0534', '0536', '0537', \
                        '0538', '0539', '0540', '0541', '0542', '0543', '0544', '0545', \
                        '0546', '0547', '0548', '0549', '0551', '0553', '0555', '0556', \
                        '0557', '0558', '0560', '0561', '0562', '0563', '0564', '0565', \
                        '0567', '0568', '0569', '0570', '0571', '0572', '0573', '0575', \
                        '0576', '0577', '0578', '0580', '0581', '0582', '0583', '0586', \
                        '0587', '0588', '0589', '0590', '0592', '0594', '0595', '0596', \
                        '0597', '0598', '0600', '0601', '0602', '0603', '0604', '0605', \
                        '0606', '0607', '0608', '0609', '0622', '0623', '0624', '0627', \
                        '0628', '0629', '0630', '0632', '0633', '0634', '0635', '0636', \
                        '0637', '0638', '0639', '0640', '0641', '0642', '0643', '0644', \
                        '0645', '0646', '0647', '0648', '0649', '0650', '0651', '0652', \
                        '0653', '0654', '0655', '0656', '0657', '0658', '0659', '0660', \
                        '0661', '0662', '0663', '0664', '0665', '0666', '0667', '0668', \
                        '0669', '0670', '0671', '0672', '0673', '0674', '0675', '0676', \
                        '0677', '0678', '0679', '0680', '0681', '0682', '0683', '0684', \
                        '0685', '0686', '0687', '0688', '0689', '0690', '0691', '0692', \
                        '0693', '0694', '0695', '0696', '0697', '0698', '0699', '0700')

DEFAULT_SXS_PREC_LIST3 = ('0701', '0702', '0703', '0704', '0705', '0706', '0707', '0708', \
                        '0709', '0710', '0711', '0712', '0713', '0714', '0715', '0716', \
                        '0717', '0718', '0719', '0720', '0721', '0722', '0723', '0724', \
                        '0725', '0726', '0727', '0728', '0729', '0730', '0731', '0732', \
                        '0733', '0734', '0735', '0736', '0737', '0738', '0739', '0740', \
                        '0741', '0742', '0743', '0744', '0745', '0746', '0747', '0748', \
                        '0749', '0750', '0751', '0752', '0753', '0754', '0755', '0756', \
                        '0757', '0758', '0759', '0760', '0761', '0762', '0763', '0764', \
                        '0765', '0766', '0767', '0768', '0769', '0770', '0771', '0772', \
                        '0773', '0774', '0775', '0776', '0777', '0778', '0779', '0780', \
                        '0781', '0782', '0783', '0784', '0785', '0786', '0787', '0788', \
                        '0789', '0790', '0791', '0792', '0793', '0794', '0795', '0796', \
                        '0797', '0798', '0799', '0800', '0801', '0802', '0803', '0804', \
                        '0805', '0806', '0807', '0808', '0809', '0810', '0811', '0812', \
                        '0813', '0814', '0815', '0816', '0817', '0818', '0819', '0820', \
                        '0821', '0822', '0823', '0824', '0825', '0826', '0827', '0828', \
                        '0829', '0830', '0831', '0832', '0833', '0834', '0835', '0836', \
                        '0837', '0838', '0839', '0840', '0841', '0842', '0843', '0844', \
                        '0845', '0846', '0847', '0848', '0849', '0850', '0851', '0852', \
                        '0853', '0854', '0855', '0856', '0857', '0858', '0859', '0860', \
                        '0861', '0862', '0863', '0864', '0865', '0866', '0867', '0868', \
                        '0869', '0870', '0871', '0872', '0873', '0874', '0875', '0876', \
                        '0877', '0878', '0879', '0880', '0881', '0882', '0883', '0884', \
                        '0885', '0886', '0887', '0888', '0889', '0890', '0891', '0892', \
                        '0893', '0894', '0895', '0896', '0897', '0898', '0899', '0900', \
                        '0901', '0902', '0903', '0904', '0905', '0906', '0907', '0908', \
                        '0909', '0910', '0911', '0912', '0913', '0914', '0915', '0916', \
                        '0917', '0918', '0919', '0920', '0921', '0922', '0923', '0924', \
                        '0925', '0926', '0927', '0928', '0929', '0930', '0931', '0932', \
                        '0933', '0934', '0935', '0936', '0937', '0938', '0939', '0940', \
                        '0941', '0942', '0943', '0944', '0945', '0946', '0947', '0948', \
                        '0949', '0950', '0951', '0952', '0953', '0954', '0955', '0956', \
                        '0957', '0958', '0959', '0960', '0961', '0962', '0963', '0964', \
                        '0965', '0966', '0967', '0968', '0969', '0970', '0971', '0972', \
                        '0973', '0974', '0975', '0976', '0977', '0978', '0979', '0980', \
                        '0981', '0982', '0983', '0984', '0985', '0986', '0987', '0988', \
                        '0989', '0990', '0991', '0992', '0993', '0994', '0995', '0996', '0997', '0998', '0999')

DEFAULT_SXS_LISTV2_SMALL_ECC = ("0083", "0087", "0091", "0100", "0105", \
        "0106", "0108", "1136", "1149", "1169")
DEFAULT_SXS_LISTV3_NOECC_ALL = list(DEFAULT_NOSPIN_SXS_LIST3) + DEFAULT_SPIN_LIST2 + DEFAULT_SPIN_LIST3

DEFAULT_SXS_LISTV2_ECC_ALL = DEFAULT_NOSPIN_ECC_SXS_LIST + list(DEFAULT_SXS_LISTV2_SMALL_ECC)
DEFAULT_SXS_LISTV2_ALL = list(DEFAULT_SXS_LISTV2_NOECC_ALL) + DEFAULT_SXS_LISTV2_ECC_ALL
DEFAULT_SXS_LISTV3_ALL = DEFAULT_SXS_LISTV3_NOECC_ALL + DEFAULT_NOSPIN_ECC_SXS_LIST + DEFAULT_SPIN_ECC_SXS_LIST
DEFAULT_SXS_TEST = ("0180", "0182", "0301", "1456", "1421", "1484", "1433", "1452", "1492")

DEFAULT_ECC_ORBIT_DICT = \
    {'1355': (0.004982474480554579, -0.047443386044795595), \
    '1356': (0.0056580808118833256, 0.08271456306292585), \
    '1357': (0.007304079904232525, 0.08528973144650712), \
    '1358': (0.00746482214867592, 0.08365671102289254), \
    '1359': (0.007567461533393136, 0.08300793298841305), \
    '1360': (0.008534403435871762, 0.11449729385994314), \
    '1361': (0.008607563846807215, 0.11487532865639345), \
    '1362': (0.009806838472454353, 0.14646921022126416), \
    '1363': (0.009874606530083216, 0.14657165625512833), \
    '1364': (0.006341210531506626, 0.03652214566519844), \
    '1365': (0.006607426390976184, 0.050438792176445285), \
    '1366': (0.007357180575296985, 0.08037157448541203), \
    '1367': (0.007431949523653447, 0.07975001429724911), \
    '1368': (0.007501242160002988, 0.07856666903771614), \
    '1369': (0.009552064862254801, 0.14310498752386835), \
    '1370': (0.009720520505439721, 0.13913459163680678), \
    '1371': (0.006578830823265621, 0.04651473812828878), \
    '1372': (0.007342398469696525, 0.07632885995339972), \
    '1373': (0.0074003276479797245, 0.07554466708583982), \
    '1374': (0.009490501724885247, 0.1351604864769108), \
    '0321': (0.00495627369014029, -0.03966133045577777), \
    '0322': (0.006529906502466181, 0.047910621269570984), \
    '0323': (0.00720188903957925, 0.077777737056339), \
    '0324': (0.009345229207398708, 0.1426120892563765),
    '0083': (0.003549489210643809, -0.017980516268702704), \
    '0087': (0.003990461221290488, 0.01411439926830114), \
    '0091': (0.003700936054887309, 0.014486146716063955), \
    '0100': (0.003931780323450942, -0.013386668809693431), \
    '0105': (0.004230923902269985, 0.012990800523233433), \
    '0106': (0.005609440389034302, -0.017881066301494417), \
    '0108': (0.005123482863654472, -0.017773498091058357), \
    '1136': (0.005153206159454922, -0.05285021734273754), \
    '1149': (0.005074593468011517, -0.032669231720378306), \
    '1169': (0.004955708621134149, 0.026440278484639715)}

DEFAULT_ECC_ORBIT_DICT_V5 = \
    {'1355': (0.004982474480554579, -0.064), \
    '1356': (0.0056580808118833256,  0.13), \
    '1357': (0.007304079904232525,0.10), \
    '1358': (0.00746482214867592, 0.10), \
    '1359': (0.007567461533393136, 0.11), \
    '1360': (0.008534403435871762,0.19), \
    '1361': (0.008607563846807215,  0.20), \
    '1362': (0.009806838472454353, 0.23), \
    '1363': (0.009874606530083216, 0.23), \
    '1364': (0.006341210531506626, 0.06), \
    '1365': (0.006607426390976184, 0.07), \
    '1366': (0.007357180575296985, 0.1), \
    '1367': (0.007431949523653447, 0.11), \
    '1368': (0.007501242160002988, 0.11), \
    '1369': (0.009552064862254801, 0.23), \
    '1370': (0.009720520505439721, 0.22), \
    '1371': (0.006578830823265621, 0.05), \
    '1372': (0.007342398469696525, 0.12), \
    '1373': (0.0074003276479797245, 0.12), \
    '1374': (0.009490501724885247, 0.22)}
import sys
def plot_SXSlist_toBash(name, LIST, nCol = 6):
    sys.stderr.write(f'{name}=(')
    for i, SXSnum in enumerate(LIST):
        if (i+1) % int(nCol) == 0:
            sys.stderr.write('\\\n\t')
        sys.stderr.write(f'\"{SXSnum}\" ')
    sys.stderr.write(')\n\n')
    return

SXSEv1ECCDICT = {}
SXSEv1ECCDICT['1355'] = 0.288342400000001
SXSEv1ECCDICT['1356'] = 0.34175168000000034
SXSEv1ECCDICT['1357'] = 0.4484716800000005
SXSEv1ECCDICT['1358'] = 0.4446604800000006
SXSEv1ECCDICT['1359'] = 0.4543040000000008
SXSEv1ECCDICT['1360'] = 0.5262367999999992
SXSEv1ECCDICT['1361'] = 0.5431647999999991
SXSEv1ECCDICT['1362'] = 0.5985299199999988
SXSEv1ECCDICT['1363'] = 0.5902309333333321
SXSEv1ECCDICT['1364'] = 0.2884197333333332
SXSEv1ECCDICT['1365'] = 0.3344268800000004
SXSEv1ECCDICT['1366'] = 0.44644783189333365
SXSEv1ECCDICT['1367'] = 0.4344627259733334
SXSEv1ECCDICT['1368'] = 0.45639130197333316
SXSEv1ECCDICT['1369'] = 0.5876
SXSEv1ECCDICT['1370'] = 0.5988751274325309
SXSEv1ECCDICT['1371'] = 0.2799424000000007
SXSEv1ECCDICT['1372'] = 0.45024435200000074
SXSEv1ECCDICT['1373'] = 0.4628845013333333
SXSEv1ECCDICT['1374'] = 0.603206399999999
