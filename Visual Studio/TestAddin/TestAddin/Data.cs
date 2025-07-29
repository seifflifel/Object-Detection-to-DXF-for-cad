using System;
using System.ComponentModel;
using CodeStack.SwEx.PMPage.Attributes;
using SolidWorks.Interop.sldworks;
using SolidWorks.Interop.swconst;


//SCALE & OFFSET UI

namespace TestAddin
{
    public class Data
    {
        [SelectionBox(swSelectType_e.swSelDATUMPLANES,SolidWorks.Interop.swconst.swSelectType_e.swSelSKETCHES)]
        [Description("Select sketch to scale")]
        [ControlAttribution(swControlBitmapLabelType_e.swBitmapLabel_SelectBoundary)] // change icon
        public IEntity Reference { get; set; }  //use this for selecting 

        [Description("Enter offset in mm")]
        [ControlAttribution(swControlBitmapLabelType_e.swBitmapLabel_LinearDistance)] // change icon
        public Double Offset { get; set; } = 0.0003;
        
        //public bool UseThinFeature { get; set; } = false;

        
    }


}
