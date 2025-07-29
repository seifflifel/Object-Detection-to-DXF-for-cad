using CodeStack.SwEx.Common.Attributes;
using CodeStack.SwEx.PMPage.Attributes;
using SolidWorks.Interop.swconst;
using System;
using System.ComponentModel;



// BOX UI

namespace TestAddin
{
    [Title("Create Box")]

    public class Data2
    {
        [Description("Enter the width of the box in mm")]
        [ControlAttribution(swControlBitmapLabelType_e.swBitmapLabel_Depth)] //change icon
        [NumberBoxOptions(swNumberboxUnitType_e.swNumberBox_Length, 0, 1000, 0.01,false,0.1,0.001,swPropMgrPageNumberBoxStyle_e.swPropMgrPageNumberBoxStyle_Thumbwheel)]
        
        public Double width { get; set; } 


        [Description("Enter the length of the box in mm")]
        [ControlAttribution(swControlBitmapLabelType_e.swBitmapLabel_Depth)] //change icon
        [NumberBoxOptions(swNumberboxUnitType_e.swNumberBox_Length, 0, 1000, 0.01, false, 0.1, 0.001, swPropMgrPageNumberBoxStyle_e.swPropMgrPageNumberBoxStyle_Thumbwheel)]

        public Double length { get; set; } 


        [Description("Enter the height of the box in mm")]
        [ControlAttribution(swControlBitmapLabelType_e.swBitmapLabel_Depth)] // change icon
        [NumberBoxOptions(swNumberboxUnitType_e.swNumberBox_Length, 0, 1000, 0.01, false, 0.1, 0.001, swPropMgrPageNumberBoxStyle_e.swPropMgrPageNumberBoxStyle_Thumbwheel)]

        public Double height { get; set; }

        
    }

}
